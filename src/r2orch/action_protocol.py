from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionSpec:
    """Provider-neutral definition of one Worker-callable Action."""

    name: str
    description: str
    parameters: dict[str, Any]
    output_schema: dict[str, Any] = field(default_factory=dict)
    examples: tuple[dict[str, Any], ...] = ()
    error_codes: tuple[str, ...] = ()
    stateful: bool = False

    def to_context_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.output_schema:
            payload["output_schema"] = self.output_schema
        if self.examples:
            payload["examples"] = list(self.examples)
        if self.error_codes:
            payload["error_codes"] = list(self.error_codes)
        if self.stateful:
            payload["stateful"] = True
        return payload


@dataclass(frozen=True)
class ActionCall:
    """One normalized Action request, independent of the model provider."""

    call_id: str
    name: str
    arguments: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }


def action_spec_from_tool(tool: Any) -> ActionSpec:
    """Build the complete public Action contract from an executable tool object."""

    return ActionSpec(
        name=str(tool.name),
        description=str(tool.description),
        parameters=dict(getattr(tool, "parameters", {}) or {}),
        output_schema=dict(getattr(tool, "output_schema", {}) or {}),
        examples=tuple(getattr(tool, "examples", ()) or ()),
        error_codes=tuple(str(value) for value in (getattr(tool, "error_codes", ()) or ())),
        stateful=bool(getattr(tool, "stateful", False)),
    )
