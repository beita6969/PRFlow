from __future__ import annotations

import ast
import json
import re
from typing import Any

# ALFWorld protocol -----------------------------------------------------
#
# The environment remains authoritative for execution.  These helpers only
# construct the bounded Worker-facing envelope used by the text executor.  In
# particular, they derive task procedures from the public task statement and
# observations; they never inspect a solution trajectory or hidden PDDL goal.
ALFWORLD_PROTOCOL_RULES = (
    "You hold at most one object; 'take' needs empty hands. Open a closed "
    "receptacle to see inside. Receptacles at your current spot need no 'go to'. "
    "Moving an object into a sinkbasin, microwave or fridge does not clean, heat or cool it."
)
ALFWORLD_PROTOCOL_PROCEDURES: dict[str, str] = {
    "pick_and_place": (
        "1. Find the object by visiting unvisited receptacles and opening closed ones. "
        "2. Take it. 3. Go to the destination. 4. Move the object there."
    ),
    "pick_two": (
        "1. Find and take one object. 2. Go to the destination and move it there. "
        "3. Find and take a second instance. 4. Move the second one to the destination."
    ),
    "clean": (
        "Find and take an X; go to a sinkbasin and run 'clean X with sinkbasin N'; "
        "then go to the destination and move X there."
    ),
    "heat": (
        "Find and take an X; go to a microwave and run 'heat X with microwave N'; "
        "then go to the destination and move X there. A hot plate means a plate."
    ),
    "cool": (
        "Find and take an X; go to a fridge and run 'cool X with fridge N'; "
        "then go to the destination and move X there."
    ),
    "look_at_obj_in_light": (
        "Find and take the object; find a desklamp (often on a desk, sidetable or "
        "dresser); go there and run 'use desklamp N' while holding the object."
    ),
}


def alfworld_protocol_task_type(task: str) -> str | None:
    """Return the public ALFWorld task family used by the ALFWorld protocol."""

    raw = str(task)
    marker = re.search(r"your task is:\s*", raw, re.I)
    text = raw[marker.end() :] if marker else raw
    text = text.casefold()
    if re.search(r"\b(?:look at|examine)\b", text) and re.search(
        r"\b(?:desk|floor)?lamp\b", text
    ):
        return "look_at_obj_in_light"
    if re.search(r"\btwo\b", text):
        return "pick_two"
    for kind, words in (("clean", r"clean"), ("heat", r"heat|hot"), ("cool", r"cool|cold")):
        if re.search(rf"\b(?:{words})\b", text):
            return kind
    if re.search(r"\b(?:put|place)\b", text):
        return "pick_and_place"
    return None


def _alfworld_public_task(state: dict[str, Any], task: str | None = None) -> str:
    value = str(task or state.get("public_task_statement", "")).strip()
    if value:
        return value
    observation = str(state.get("initial_observation", state.get("observation", "")))
    _head, marker, goal = observation.partition("Your task is:")
    return goal.strip() if marker else observation.strip()


def _alfworld_commands(state: dict[str, Any]) -> list[dict[str, str]]:
    raw = state.get("admissible_actions", state.get("admissible_commands", ()))
    if not isinstance(raw, (list, tuple)):
        return []
    return [
        (
            {"action_id": str(item.get("action_id", "")), "command": str(item.get("command", ""))}
            if isinstance(item, dict)
            else {"action_id": str(item), "command": str(item)}
        )
        for item in raw
        if (isinstance(item, dict) and str(item.get("action_id", "")).strip())
        or (isinstance(item, str) and item.strip())
    ]


def _alfworld_locations(state: dict[str, Any]) -> list[str]:
    """Extract only visible numbered receptacle/room names."""

    known: list[str] = []
    memory = state.get("factual_memory", {})
    if isinstance(memory, dict):
        for value in memory.get("visited_locations", ()):
            if str(value).strip() and str(value) not in known:
                known.append(str(value))
        for value in memory.get("location_observations", {}) if isinstance(memory.get("location_observations"), dict) else ():
            if str(value).strip() and str(value) not in known:
                known.append(str(value))
        current = str(memory.get("current_location", "")).strip()
        if current and current not in known:
            known.append(current)
    observation = str(state.get("initial_observation", ""))
    match = re.search(r"Looking quickly around you, you see (.+?)\.(?:\s|$)", observation, re.I | re.S)
    if match:
        for item in re.split(r",\s*(?:and\s+)?|\s+and\s+", match.group(1)):
            name = re.sub(r"^(?:an?|the)\s+", "", item.strip().casefold())
            if re.fullmatch(r"[a-z][a-z ]*\d+", name) and name not in known:
                known.append(name)
    initial_raw = state.get("initial_admissible_actions", state.get("initial_admissible_commands", ()))
    initial_state = {"admissible_actions": initial_raw}
    for item in [*_alfworld_commands(initial_state), *_alfworld_commands(state)]:
        command = item["command"]
        if command.startswith("go to "):
            name = command[6:].strip()
            if name and name not in known:
                known.append(name)
    return known


def _alfworld_current_location(state: dict[str, Any]) -> str:
    memory = state.get("factual_memory", {})
    if isinstance(memory, dict) and str(memory.get("current_location", "")).strip():
        return str(memory["current_location"])
    match = re.search(r"You arrive at (.+?)\.", str(state.get("observation", "")), re.I)
    return match.group(1).strip() if match else ""


def _alfworld_holding(state: dict[str, Any]) -> str:
    memory = state.get("factual_memory", {})
    interactions = memory.get("recent_interactions", ()) if isinstance(memory, dict) else ()
    holding = ""
    for item in interactions:
        if not isinstance(item, dict):
            continue
        command = str(item.get("command", ""))
        observation = str(item.get("observation", ""))
        picked = re.match(r"You pick up the (.+?) from ", observation, re.I)
        carrying = re.match(r"You are carrying:\s*(?:an?\s+|the\s+)?(.+?)\.?$", observation, re.I)
        if picked:
            holding = picked.group(1).strip()
        elif carrying:
            holding = carrying.group(1).strip()
        elif command.startswith(("move ", "put ")) or "not carrying anything" in observation.casefold():
            holding = ""
    return holding


def alfworld_protocol_location_memory(state: dict[str, Any], *, max_chars: int = 800) -> dict[str, Any]:
    """Project bounded location facts and an explicit unvisited list."""

    memory = state.get("factual_memory", {})
    memory = memory if isinstance(memory, dict) else {}
    observations = memory.get("location_observations", {})
    observations = observations if isinstance(observations, dict) else {}
    visited = [str(v) for v in memory.get("visited_locations", ()) if str(v).strip()]
    known = _alfworld_locations(state)
    current = _alfworld_current_location(state)
    value: dict[str, Any] = {
        "current": current,
        "visited_last_seen": {},
        "not_visited": [
            name for name in known if name not in visited and name.casefold() != current.casefold()
        ],
    }
    for name in visited:
        row = observations.get(name)
        text = str(row.get("observation", "")) if isinstance(row, dict) else ""
        if text:
            value["visited_last_seen"][name] = text[:160]
    limit = max(0, int(max_chars))
    while len(json.dumps(value, ensure_ascii=False)) > limit and value["visited_last_seen"]:
        value["visited_last_seen"].pop(next(iter(value["visited_last_seen"])))
    while len(json.dumps(value, ensure_ascii=False)) > limit and value["not_visited"]:
        value["not_visited"].pop()
    if len(json.dumps(value, ensure_ascii=False)) > limit and value["current"]:
        value["current"] = ""
    return {key: item for key, item in value.items() if item not in ("", {}, [])}


def alfworld_protocol_loop_note(state: dict[str, Any]) -> str:
    memory = state.get("factual_memory", {})
    recent = memory.get("recent_interactions", ()) if isinstance(memory, dict) else ()
    if len(recent) < 2:
        return ""
    last = recent[-1] if isinstance(recent[-1], dict) else {}
    command = str(last.get("command", "")).strip()
    observation = str(last.get("observation", "")).strip()
    if not command:
        return ""
    repeats = sum(
        1
        for item in recent
        if isinstance(item, dict)
        and str(item.get("command", "")).strip() == command
        and str(item.get("observation", "")).strip() == observation
    )
    if repeats < 2:
        return ""
    locations = alfworld_protocol_location_memory(state).get("not_visited", [])
    suffix = ", ".join(str(value) for value in locations[:3])
    suggestion = f"go to one you have not visited: {suffix}" if suffix else "open or take something here, or do the next procedure step"
    return f"'{command[:80]}' produced the same result {repeats} times; do something different (e.g. {suggestion})."


def alfworld_protocol_invalid_feedback(state: dict[str, Any], action_id: object, error: str = "") -> dict[str, Any]:
    """Explain a rejected state-scoped action and provide fresh examples."""

    aid = str(action_id or "").strip()
    commands = _alfworld_commands(state)
    ids = {item["action_id"] for item in commands}
    by_command = {" ".join(item["command"].casefold().split()): item for item in commands}

    # Native tool failures provide an action id, while text-only recovery can
    # provide the command itself.  Keep both forms useful: stale ids get a
    # state-scoped explanation, and a rejected command gets a concrete reason
    # derived only from public observations and the latest admissible list.
    command = ""
    if aid in ids:
        command = next(item["command"] for item in commands if item["action_id"] == aid)
    elif " ".join(aid.casefold().split()) in by_command:
        command = by_command[" ".join(aid.casefold().split())]["command"]
    elif " " in aid:
        # Text-only callers may pass the rejected command itself rather than
        # an action id.  It is still a public attempt and can be explained.
        command = aid
    if aid and aid not in ids and not command:
        reason = f"Action id '{aid}' is stale or unknown for the latest observation."
    elif command:
        reason = str(error).strip() or _alfworld_invalid_command_reason(command, state)
    else:
        reason = str(error).strip() or "Your reply has no 'Action:' line."
    return {
        "reason": reason,
        "examples": [item["command"] for item in commands[:4]],
        "instruction": "Reply with one Thought line and one Action line; copy the Action command verbatim from admissible_commands.",
    }


def _alfworld_invalid_command_reason(command: str, state: dict[str, Any]) -> str:
    """Explain a rejected public command without consulting hidden goal state."""

    command = " ".join(str(command).casefold().split())
    current = _alfworld_current_location(state)
    known = {name.casefold() for name in _alfworld_locations(state)}
    holding = _alfworld_holding(state).casefold()
    where = f"you are at {current}" if current else "you are in the middle of the room"
    match = re.fullmatch(r"go to (.+)", command)
    if match and current and match.group(1) == current.casefold():
        return f"You are already at {match.group(1)}."
    if match and match.group(1) not in known:
        return f"There is no {match.group(1)} in the known room list."
    match = re.fullmatch(r"take (.+?) from (.+)", command)
    if match:
        item, place = match.groups()
        if holding:
            return f"You are already holding {holding}; put it somewhere before taking another object."
        if current and place != current.casefold():
            return f"You are not at {place} ({where})."
        return f"There is no {item} at {place} now."
    match = re.fullmatch(r"(?:move|put) (.+?) (?:to|in|on|into|onto|in/on) (.+)", command)
    if match:
        item, place = match.groups()
        if not holding:
            return f"You are not holding anything; take {item} first."
        if holding != item:
            return f"You are holding {holding}, not {item}."
        if current and place != current.casefold():
            return f"You are not at {place} ({where})."
        return f"You cannot put {item} in {place} now; open it first if it is closed."
    match = re.fullmatch(r"(?:clean|heat|cool) (.+?) with (.+)", command)
    if match:
        _verb, item, place = match.groups()
        if holding != item:
            return f"You are not holding {item}."
        if current and place != current.casefold():
            return f"You are not at {place} ({where})."
    return f"'{command[:80]}' is not one of admissible_actions here."


def alfworld_protocol_parse(text: str) -> tuple[str | None, str]:
    """Parse a Worker Thought + Action envelope without executing it."""

    # Strip presentation around labels, not inside command/JSON identifiers.
    # In particular, alfworld_step and action_id must keep their underscores.
    lines = [re.sub(r"^\s*[#>]+\s*", "", line).strip() for line in str(text).splitlines()]
    for index, line in enumerate(lines):
        match = re.match(r"^[*_`]*action\s*\d*\s*[*_`]*[:：]\s*[*_`]*\s*(.*)$", line, re.I)
        if match:
            command = match.group(1).strip() or next((rest for rest in lines[index + 1 :] if rest), "")
            command = command.strip("`")
            thought = " ".join(
                re.sub(r"^[*_`]*thought\s*\d*\s*[*_`]*[:：]\s*[*_`]*\s*", "", before, flags=re.I)
                for before in lines[:index]
                if before
            )
            return command or None, thought[:200]
    match = re.search(r"\baction\s*\d*\s*[:：]\s*(.+)$", str(text), re.I | re.M)
    if match:
        before = str(text)[: match.start()]
        thought_lines = [
            re.sub(r"^thought\s*\d*\s*[:：]\s*", "", line.strip(), flags=re.I)
            for line in before.splitlines()
            if line.strip()
        ]
        return match.group(1).strip(), " ".join(thought_lines)[:200]
    return None, ""


def _alfworld_prose_action_in_example(prefix: str) -> bool:
    """Inspect the Action's region, without filtering words in ordinary Thought."""
    fence = ""
    triple_quote = ""
    for line in prefix.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if not fence:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = ""
        elif not fence:
            for match in re.finditer(r"(?<!\\)(?:\"\"\"|''')", line):
                value = match.group(0)
                if not triple_quote:
                    triple_quote = value
                elif triple_quote == value:
                    triple_quote = ""
    paragraph = re.split(r"\n\s*\n", prefix)[-1]
    return bool(fence or triple_quote or re.search(
        r"(?im)^\s*(?:Thought\s*[:：]\s*)?"
        r"(?:example|template|schema|sample|source(?:\s+code)?|for example"
        r"|here is (?:an? )?(?:example|template|sample)|示例|模板|源码)\s*[:：.]\s*$",
        paragraph,
    ))


def alfworld_protocol_select_action(text: str, state: dict[str, Any]) -> tuple[str | None, str, str]:
    """Map a parsed command to the current action id (exact or word match)."""

    command, thought = alfworld_protocol_parse(text)
    if command is None:
        return None, thought, "no_action"
    # Earlier Thought text can describe or quote "the action: ...". Only
    # inspect the actual Action section for a second action envelope.
    action_start = re.search(
        r"^\s*[#>]*\s*[*_`]*action\s*\d*\s*[*_`]*[:：]",
        str(text), re.I | re.M,
    )
    action_section = str(text)[action_start.start():] if action_start else str(text)
    action_lines = re.findall(
        r"\baction\s*\d*\s*[*_`]*[:：]",
        action_section, re.I | re.M,
    )
    if len(action_lines) > 1:
        return None, thought, "invalid"
    choices = _alfworld_commands(state)
    if re.match(r"^alfworld_step\b", command):
        prose_call = re.fullmatch(
            r"alfworld_step\s+with\s+action_id\s+(['\"])([^'\"\r\n]*)\1", command,
        )
        if prose_call:
            # This observed spelling is an explicit Action, not executable
            # Python. Inspect the complete section so an example, quoted source
            # or second trailing request cannot become an environment action.
            prefix = str(text)[:action_start.start()] if action_start else str(text)
            if (action_start is None or len(action_section.strip().splitlines()) != 1
                    or re.search(r"(?m)^\s*[#>]*>", action_section)
                    or _alfworld_prose_action_in_example(prefix)):
                return None, thought, "invalid"
            live = [item for item in choices if item["action_id"] == prose_call.group(2)]
            if len(live) != 1 or not prose_call.group(2).strip():
                return None, thought, "invalid"
            return live[0]["action_id"], thought, "action_id"
        # Inspect the syntax, never execute it. Both quote styles and the named
        # argument spelling resolve only a single literal, live action_id.
        try:
            call = ast.parse(command, mode="eval").body
        except (SyntaxError, ValueError, RecursionError):
            return None, thought, "invalid"
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id != "alfworld_step"):
            return None, thought, "invalid"
        if len(call.args) == 1 and not call.keywords and isinstance(call.args[0], ast.Dict):
            argument = call.args[0]
            # AST retains duplicate keys, which must not silently override.
            if (len(argument.keys) != 1 or not isinstance(argument.keys[0], ast.Constant)
                    or argument.keys[0].value != "action_id"):
                return None, thought, "invalid"
            action_id = argument.values[0]
        elif not call.args and len(call.keywords) == 1 and call.keywords[0].arg == "action_id":
            action_id = call.keywords[0].value
        else:
            return None, thought, "invalid"
        if not isinstance(action_id, ast.Constant) or not isinstance(action_id.value, str):
            return None, thought, "invalid"
        live = [item for item in choices if item["action_id"] == action_id.value]
        if len(live) != 1:
            return None, thought, "invalid"
        return live[0]["action_id"], thought, "action_id"
    normalized = " ".join(command.casefold().strip("`'\".").split())
    by_norm = {" ".join(item["command"].casefold().split()): item for item in choices}
    if normalized in by_norm:
        return by_norm[normalized]["action_id"], thought, "exact"
    for key, item in sorted(by_norm.items(), key=lambda pair: len(pair[0]), reverse=True):
        if key and re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", normalized):
            return item["action_id"], thought, "word"
    return None, thought, "invalid"


def alfworld_protocol_prompt(
    state: dict[str, Any], *, task: str | None = None, rejected_reply: str | None = None,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    """Build the bounded ALFWorld-protocol Worker context."""

    public_task = _alfworld_public_task(state, task)
    kind = alfworld_protocol_task_type(public_task)
    memory = alfworld_protocol_location_memory(state)
    current = _alfworld_current_location(state)
    value: dict[str, Any] = {
        "protocol": "ALFWorld",
        "task": public_task,
        "procedure": ALFWORLD_PROTOCOL_RULES + (
            " Steps: " + ALFWORLD_PROTOCOL_PROCEDURES[kind] if kind in ALFWORLD_PROTOCOL_PROCEDURES else ""
        ),
        "state": {"location": current or "the middle of the room", "holding": _alfworld_holding(state) or "nothing"},
        "locations": memory,
        "observation": str(state.get("observation", "")),
        "admissible_commands": [item["command"] for item in _alfworld_commands(state)],
        "instruction": "Reply with exactly one 'Thought: <one or two short sentences>' line and one 'Action: <command>' line. Copy the Action command verbatim from admissible_commands.",
    }
    warning = alfworld_protocol_loop_note(state)
    if warning:
        value["state"]["warning"] = warning
    if rejected_reply is not None:
        feedback = alfworld_protocol_invalid_feedback(state, "", rejection_reason or "")
        value["rejected_reply"] = str(rejected_reply)[:240]
        value["error"] = f"{feedback['reason']} The game did not run it. Examples: " + ", ".join(feedback["examples"])
    return {key: item for key, item in value.items() if item not in ("", {}, [])}


def alfworld_protocol_instruction() -> str:
    return (
        "The ALFWorld protocol is active. For every environment decision, reason briefly with "
        "'Thought:' and then choose exactly one admissible command in 'Action:'. The runtime "
        "executes only alfworld_step with an action_id from the latest observation; do not invent "
        "ids or claim that a command ran. Use the task-type procedure, visible location memory, "
        "and loop warning. If an action is rejected, use the stated reason and choose a fresh "
        "admissible command."
    )
