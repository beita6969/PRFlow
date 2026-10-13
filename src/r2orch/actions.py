from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .contracts import RelationType

POLICY_PARSE_ERROR_CODES = frozenset({
    "director_json_syntax_error", "director_json_object_missing",
    "director_multiple_action_objects", "director_action_schema_error",
})

# Wire fields, not suggested actions or graph edits. SET_PROMPT uses the
# structured form advertised to Director; parse() also accepts a free-form prompt,
# which the Canvas rejects.
ACTION_FIELDS = {
    "add_agent": ("action",),
    "set_prompt": ("action", "target", "role", "objective", "scope", "expected_output"),
    "set_model": ("action", "target", "runtime_route"),
    "set_layer": ("action", "target", "layer"),
    "consider_relation": ("action", "source", "target"),
    "set_relation": ("action", "source", "target", "relation"),
    "remove_relation": ("action", "source", "target", "relation"),
    "delete_agent": ("action", "target"),
    "set_output": ("action", "target"),
    "finish": ("action",),
    "run_agent": ("action", "target"),
}

UNIFIED_ACTION_FIELDS = {
    **{key: value for key, value in ACTION_FIELDS.items() if key != "set_output"},
    "set_prompt": (*ACTION_FIELDS["set_prompt"], "result_scope"),
    "finish": ("action", "target"),
}

_POLICY_JSON_KEYS = frozenset(
    {
        key
        for fields in (
            *ACTION_FIELDS.values(),
            *UNIFIED_ACTION_FIELDS.values(),
            ("action_type", "agent_id", "model_route", "runtime", "thought"),
            ("model_id", "layer_id", "layer_number"),
        )
        for key in fields
    }
)


class ActionType(StrEnum):
    ADD_AGENT = "add_agent"
    SET_PROMPT = "set_prompt"
    SET_MODEL = "set_model"
    SET_LAYER = "set_layer"
    CONSIDER_RELATION = "consider_relation"
    # The Canvas applies the edge chosen for a ConsiderRelation proposal with these two
    # (resolve_relation_choice); they are never in allowed_actions, so the Director cannot emit them.
    SET_RELATION = "set_relation"
    REMOVE_RELATION = "remove_relation"
    DELETE_AGENT = "delete_agent"
    SET_OUTPUT = "set_output"
    FINISH = "finish"
    RUN_AGENT = "run_agent"
    INVALID = "invalid"


class PromptRevisionBasis(StrEnum):
    UPSTREAM_ARTIFACT_CHANGED = "upstream_artifact_changed"
    PEER_ARTIFACT_CHANGED = "peer_artifact_changed"
    TOOL_ERROR = "tool_error"
    UNRESOLVED_ISSUE = "unresolved_issue"
    PROTOCOL_FAILURE = "protocol_failure"
    STRUCTURAL_ROLE_CHANGE = "structural_role_change"
    CONTROLLER_REPAIR = "controller_repair"


@dataclass
class CanvasAction:
    """One atomic graph edit, following FlowSteer's one-action-per-turn contract."""

    action_type: ActionType
    agent_id: str | None = None
    target: str | None = None
    source: str | None = None
    prompt: str | None = None
    role: str | None = None
    objective: str | None = None
    scope: str | None = None
    expected_output: str | None = None
    result_scope: str | None = None
    revision_basis: PromptRevisionBasis | None = None
    evidence_agent_ids: tuple[str, ...] = ()
    layer: int | None = None
    relation: RelationType | None = None
    reasoning: str | None = None
    runtime_route: str | None = None
    expected_version: int | None = None
    raw_text: str = ""
    parse_error: str | None = None
    parse_error_code: str | None = None
    parse_error_details: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.action_type is not ActionType.INVALID and self.parse_error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_type"] = self.action_type.value
        payload["relation"] = self.relation.value if self.relation else None
        payload["revision_basis"] = self.revision_basis.value if self.revision_basis else None
        payload["evidence_agent_ids"] = list(self.evidence_agent_ids)
        return payload


@dataclass(frozen=True)
class ParsedPolicyAction:
    """Boundary between immutable provider text and one Canvas action."""

    action: CanvasAction
    action_text: str | None
    character_span: tuple[int, int] | None
    candidate_count: int
    repair_applied: bool = False


class ActionParser:
    """Parse a single JSON action."""

    def parse(self, text: str) -> CanvasAction:
        if not text or not text.strip():
            return self._invalid(text or "", "empty action")
        payload = self._json_payload(text)
        if payload is not None:
            return self._from_payload(payload, text)
        return self._invalid(text, "expected one JSON object")

    def parse_policy_output(self, text: str) -> ParsedPolicyAction:
        """Accept exactly one complete, valid JSON action from a policy response.

        Reasoning outside the JSON object is retained in the trajectory but is never
        interpreted as another action.
        """

        raw = str(text or "")
        candidates = list(_top_level_json_objects(raw))
        if len(candidates) != 1:
            # A partial object is a syntax error, not an invented empty action.
            # Diagnose the original bytes/characters; never fix or execute them.
            start = raw.find("{")
            if not candidates and start >= 0:
                try:
                    json.loads(raw[start:])
                except json.JSONDecodeError as exc:
                    return self._policy_json_error(raw, exc, start, None, 0)
            invalid = self._invalid(
                raw, f"expected exactly one JSON action; found {len(candidates)}"
            )
            invalid.parse_error_code = (
                "director_multiple_action_objects" if candidates else "director_json_object_missing"
            )
            invalid.parse_error_details = {
                "object_count": len(candidates),
                "message": "The action channel must contain one JSON object, without examples or extra actions.",
            }
            return ParsedPolicyAction(invalid, None, None, len(candidates))
        start, end, candidate = candidates[0]
        repair_applied = False
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            repaired_candidate = _repair_json_syntax(candidate, allowed_keys=_POLICY_JSON_KEYS)
            repair_candidates = [repaired_candidate]
            escaped_candidate = _repair_json_string_backslashes(repaired_candidate)
            if escaped_candidate != repaired_candidate:
                repair_candidates.append(escaped_candidate)
            if all(item == candidate for item in repair_candidates):
                count = 0 if end == len(raw) and not candidate.rstrip().endswith("}") else 1
                return self._policy_json_error(
                    raw, exc, start, (start, end) if count else None, count
                )
            payload = None
            parsed_candidate = None
            for repair_candidate in repair_candidates:
                try:
                    payload = json.loads(repair_candidate)
                except (TypeError, ValueError):
                    continue
                parsed_candidate = repair_candidate
                break
            if parsed_candidate is None:
                return self._policy_json_error(raw, exc, start, (start, end), 1)
            candidate = parsed_candidate
            repair_applied = True
        if not isinstance(payload, dict):
            invalid = self._invalid(raw, "the single JSON action must be an object")
            return ParsedPolicyAction(invalid, None, (start, end), 1, repair_applied)
        action = self._from_payload(payload, candidate)
        if not action.valid:
            name = str(payload.get("action", payload.get("action_type", ""))).lower()
            action.parse_error_code = "director_action_schema_error"
            action.parse_error_details = {
                "action": name, "message": action.parse_error,
                "received_fields": sorted(payload),
                    "structured_action_fields": list(UNIFIED_ACTION_FIELDS.get(name, ("action",))),
            }
            if repair_applied:
                action.parse_error_details["syntax_repair_applied"] = True
            if "relations" in payload and name == "consider_relation":
                action.parse_error_details["message"] = (
                    "relations is a Canvas candidate list, not an action field. "
                    "A single relation action uses top-level source and target. "
                    "The action must still appear in allowed_actions."
                )
        return ParsedPolicyAction(
            action,
            candidate if action.valid else None,
            (start, end),
            1,
            repair_applied,
        )

    @staticmethod
    def _policy_json_error(raw, exc, start, span, count) -> ParsedPolicyAction:
        offset = start + exc.pos
        details = {
            "message": exc.msg,
            "line": raw.count("\n", 0, offset) + 1,
            "column": offset - raw.rfind("\n", 0, offset),
            "raw_action_offset": offset,
            "fragment": raw[max(0, offset - 24):offset + 48],
            "object_count": count,
        }
        if exc.msg.startswith("Invalid \\escape"):
            details["hint"] = (
                r'Inside JSON string values, a literal backslash needs two backslashes: '
                r'"Use \\omega". Plain words such as "omega" also avoid LaTeX escaping.'
            )
        invalid = CanvasAction(
            action_type=ActionType.INVALID, raw_text=raw,
            parse_error=f"JSON syntax error at line {details['line']}, column {details['column']}: {exc.msg}",
            parse_error_code="director_json_syntax_error", parse_error_details=details,
        )
        return ParsedPolicyAction(invalid, None, span, count)

    def _from_payload(self, payload: dict[str, Any], raw_text: str) -> CanvasAction:
        action_name = str(payload.get("action", payload.get("action_type", ""))).lower()
        # Directors sometimes attach routing/layer bookkeeping to SET_PROMPT.
        # Those fields do not change the prompt and are not legal unified
        # action fields; retain the prompt while leaving model selection to the
        # following SET_MODEL action.
        if action_name == "set_prompt":
            payload = {
                key: value
                for key, value in payload.items()
                if key not in {"model_id", "model_route", "layer_id", "layer_number"}
            }
        try:
            action_type = ActionType(action_name)
        except ValueError:
            return self._invalid(raw_text, f"unknown action: {action_name or 'missing'}")
        relation = payload.get("relation")
        try:
            relation_type = RelationType(str(relation).lower()) if relation is not None else None
        except ValueError:
            return self._invalid(raw_text, f"unknown relation: {relation}")
        raw_revision_basis = payload.get("revision_basis")
        try:
            revision_basis = (
                PromptRevisionBasis(str(raw_revision_basis).lower())
                if raw_revision_basis is not None
                else None
            )
        except ValueError:
            return self._invalid(
                raw_text,
                f"unknown prompt revision_basis: {raw_revision_basis}",
            )
        raw_evidence_ids = payload.get("evidence_agent_ids", ())
        if raw_evidence_ids is None:
            evidence_agent_ids: tuple[str, ...] = ()
        elif isinstance(raw_evidence_ids, (list, tuple)):
            evidence_agent_ids = tuple(
                dict.fromkeys(
                    value for item in raw_evidence_ids if (value := _clean(item)) is not None
                )
            )
        else:
            return self._invalid(raw_text, "evidence_agent_ids must be an array")
        layer = payload.get("layer")
        try:
            parsed_layer = int(layer) if layer is not None else None
        except (TypeError, ValueError):
            return self._invalid(raw_text, "layer must be an integer")
        expected_version = payload.get("expected_version")
        try:
            parsed_expected_version = (
                int(expected_version) if expected_version is not None else None
            )
        except (TypeError, ValueError):
            return self._invalid(raw_text, "expected_version must be an integer")
        action = CanvasAction(
            action_type=action_type,
            agent_id=_clean(payload.get("agent_id")),
            target=_clean(payload.get("target")),
            source=_clean(payload.get("source")),
            prompt=_clean(payload.get("prompt")),
            role=_clean(payload.get("role")),
            objective=_clean(payload.get("objective")),
            scope=_clean(payload.get("scope")),
            expected_output=_clean(payload.get("expected_output")),
            result_scope=_clean(payload.get("result_scope")),
            revision_basis=revision_basis,
            evidence_agent_ids=evidence_agent_ids,
            layer=parsed_layer,
            relation=relation_type,
            reasoning=_clean(payload.get("reasoning")),
            runtime_route=_clean(
                payload.get("runtime_route", payload.get("model_route", payload.get("runtime")))
            ),
            expected_version=parsed_expected_version,
            raw_text=raw_text,
        )
        error = self._required_field_error(action)
        if action.action_type is ActionType.SET_OUTPUT:
            error = "set_output is not part of the unified protocol; configure result_scope and finish(target)"
        elif action.action_type in {ActionType.FINISH, ActionType.RUN_AGENT} and not action.target:
            error = f"{action.action_type.value} requires target"
        elif action.action_type is ActionType.SET_PROMPT and action.result_scope not in {"subtask", "task_result"}:
            error = "set_prompt requires result_scope=subtask or task_result"
        if error:
            return self._invalid(raw_text, error, reasoning=action.reasoning)
        return action

    @staticmethod
    def _json_payload(text: str) -> dict[str, Any] | None:
        without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = [without_think.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", without_think, re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1))
        first, last = without_think.find("{"), without_think.rfind("}")
        if first >= 0 and last > first:
            candidates.append(without_think[first : last + 1])
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError):
                try:
                    payload = json.loads(_repair_json_string_backslashes(candidate))
                except (TypeError, ValueError):
                    continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _required_field_error(action: CanvasAction) -> str | None:
        if action.action_type is ActionType.ADD_AGENT:
            return None
        if (
            action.action_type
            in {
                ActionType.SET_PROMPT,
                ActionType.SET_MODEL,
                ActionType.SET_LAYER,
                ActionType.SET_OUTPUT,
            }
            and not action.target
        ):
            return f"{action.action_type.value} requires target"
        if action.action_type is ActionType.SET_PROMPT:
            structured = (
                action.role,
                action.objective,
                action.scope,
                action.expected_output,
            )
            if not action.prompt and not all(structured):
                return "set_prompt requires role, objective, scope, and expected_output"
        if action.action_type is ActionType.SET_LAYER and action.layer is None:
            return "set_layer requires layer"
        if action.action_type is ActionType.SET_MODEL and not action.runtime_route:
            return "set_model requires runtime_route"
        if action.runtime_route is not None and action.action_type is not ActionType.SET_MODEL:
            return "runtime_route is only valid in set_model"
        if action.action_type is ActionType.CONSIDER_RELATION and (
            not action.source or not action.target
        ):
            return "consider_relation requires source and target"
        if action.action_type in {
            ActionType.SET_RELATION,
            ActionType.REMOVE_RELATION,
        } and (not action.source or not action.target or action.relation is None):
            return f"{action.action_type.value} requires source, target, and relation"
        if action.action_type is ActionType.DELETE_AGENT and not (action.target or action.agent_id):
            return "delete_agent requires target or agent_id"
        return None

    @staticmethod
    def _invalid(raw_text: str, error: str, *, reasoning: str | None = None) -> CanvasAction:
        return CanvasAction(
            action_type=ActionType.INVALID,
            raw_text=raw_text,
            parse_error=error,
            reasoning=reasoning,
        )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _top_level_json_objects(text: str):
    """Yield complete JSON-object candidates and their exact character spans."""

    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            return
        depth = 0
        in_string = False
        escaped = False
        cursor = start
        while cursor < len(text):
            character = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    yield start, end, text[start:end]
                    index = end
                    break
            cursor += 1
        else:
            # Preserve a single truncated object as a candidate so the
            # bounded syntax-repair path can recover a missing final closer.
            # An unfinished string is still rejected by the repair helper.
            if depth > 0:
                yield start, len(text), text[start:]
            return


def _repair_json_string_backslashes(value: str) -> str:
    """Escape model-emitted LaTeX slashes only after strict JSON parsing fails.

    A retry is intentionally narrow: outside JSON strings nothing changes, and
    quote, slash, backslash and valid Unicode escapes retain JSON semantics.
    Other string escapes become literal backslashes. This recovers ``\angle``
    and also prevents ``\frac`` from silently becoming a form-feed escape in
    a candidate that already required repair.
    """

    repaired: list[str] = []
    in_string = False
    index = 0
    while index < len(value):
        character = value[index]
        if character == '"':
            backslashes = 0
            cursor = len(repaired) - 1
            while cursor >= 0 and repaired[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
            repaired.append(character)
            index += 1
            continue
        if character != "\\" or not in_string or index + 1 >= len(value):
            repaired.append(character)
            index += 1
            continue
        following = value[index + 1]
        unicode_escape = (
            following == "u"
            and index + 5 < len(value)
            and all(char in "0123456789abcdefABCDEF" for char in value[index + 2 : index + 6])
        )
        if following in {'"', "\\", "/"} or unicode_escape:
            repaired.append(character)
        else:
            repaired.extend(("\\", "\\"))
        index += 1
    return "".join(repaired)


def _repair_json_syntax(value: str, *, allowed_keys: frozenset[str]) -> str:
    """Quote allowlisted bare object keys without touching string contents.

    Some model responses are almost JSON, for example
    ``{"action":"set_prompt", result_scope:"task_result"}``.  This narrow
    lexical repair makes that representation parseable while leaving unknown
    keys and all values unchanged.  It never evaluates arbitrary text.
    """

    repaired: list[str] = []
    index = 0
    in_string = False
    escaped = False
    changed = False
    length = len(value)
    while index < length:
        character = value[index]
        if in_string:
            boundary = _misquoted_field_boundary(
                value, index, allowed_keys=allowed_keys
            )
            if character == "," and boundary is not None:
                key, end = boundary
                repaired.extend(('"', ',', '"', key, '"', ':'))
                index = end
                changed = True
                continue
            if ord(character) < 0x20:
                repaired.append({"\n": r"\n", "\r": r"\r", "\t": r"\t"}.get(
                    character, f"\\u{ord(character):04x}"
                ))
                changed = True
                index += 1
                continue
            repaired.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            key_boundary = _missing_field_colon(
                value, index, allowed_keys=allowed_keys
            )
            if key_boundary is not None:
                end, whitespace, key = key_boundary
                repaired.append(value[index:end])
                repaired.append(whitespace)
                repaired.append(":")
                index = end + len(whitespace) + 1
                changed = True
                continue
            in_string = True
            repaired.append(character)
            index += 1
            continue
        if character in "{,":
            cursor = index + 1
            while cursor < length and value[cursor].isspace():
                cursor += 1
            key_start = cursor
            if cursor < length and (value[cursor].isalpha() or value[cursor] == "_"):
                cursor += 1
                while cursor < length and (value[cursor].isalnum() or value[cursor] in "_-"):
                    cursor += 1
                key = value[key_start:cursor]
                whitespace_end = cursor
                while cursor < length and value[cursor].isspace():
                    cursor += 1
                if key in allowed_keys and cursor < length and value[cursor] == ":":
                    repaired.append(character)
                    repaired.append(value[index + 1 : key_start])
                    repaired.extend(('"', key, '"'))
                    repaired.append(value[whitespace_end:cursor])
                    repaired.append(":")
                    index = cursor + 1
                    changed = True
                    continue
        repaired.append(character)
        index += 1
    result = "".join(repaired) if changed else value
    # A provider can truncate the final object after a complete quoted value,
    # leaving only the closing brace missing (for example
    # ``{"action":"finish"``).  Add only delimiters that are provably
    # unmatched outside strings; never guess an unfinished string or a
    # trailing comma.  This keeps the repair bounded and syntax-only.
    result = _append_missing_json_closers(result)
    return result if result != value else value


def _append_missing_json_closers(value: str) -> str:
    """Append unmatched object/array closers when the lexical state is safe."""

    stack: list[str] = []
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append("]" if character == "[" else "}")
        elif character in "]}":
            if not stack or stack[-1] != character:
                return value
            stack.pop()
    if in_string or not stack or value.rstrip().endswith((",", ":")):
        return value
    return value + "".join(reversed(stack))


def _misquoted_field_boundary(
    value: str, start: int, *, allowed_keys: frozenset[str]
) -> tuple[str, int] | None:
    """Find `,'scope":` or `,"scope":` while already inside a value string."""

    cursor = start + 1
    if cursor >= len(value):
        return None
    if value[cursor] == "'":
        cursor += 1
    key_start = cursor
    if cursor >= len(value) or not (value[cursor].isalpha() or value[cursor] == "_"):
        return None
    cursor += 1
    while cursor < len(value) and (value[cursor].isalnum() or value[cursor] in "_-"):
        cursor += 1
    key = value[key_start:cursor]
    if key not in allowed_keys or cursor >= len(value) or value[cursor] != '"':
        return None
    cursor += 1
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    if cursor >= len(value) or value[cursor] != ":":
        return None
    return key, cursor + 1


def _missing_field_colon(
    value: str, start: int, *, allowed_keys: frozenset[str]
) -> tuple[int, str, str] | None:
    """Find an allowlisted object key followed by a comma instead of a colon."""

    cursor = start + 1
    key_start = cursor
    escaped = False
    while cursor < len(value):
        character = value[cursor]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            key = value[key_start:cursor]
            if key not in allowed_keys:
                return None
            after = cursor + 1
            whitespace_start = after
            while after < len(value) and value[after].isspace():
                after += 1
            if after >= len(value) or value[after] != ",":
                return None
            next_value = after + 1
            while next_value < len(value) and value[next_value].isspace():
                next_value += 1
            if next_value >= len(value) or value[next_value] != '"':
                return None
            return cursor + 1, value[whitespace_start:after], key
        cursor += 1
    return None
