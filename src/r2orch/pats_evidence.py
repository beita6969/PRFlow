"""Public, bounded PATS process evidence from the actual Canvas trace schema.

Like upstream PATS evidence cards, retain executable actions and observations,
compare all credible sibling attempts, and remove model reasoning. Private
verification data is excluded before this extractor and again by field selection.
Sampling affects representative text only: statistics always use the whole trace.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .actions import ActionType

_PRIVATE_KEYS = {
    "answer",
    "answers",
    "reference",
    "references",
    "reference_answer",
    "reference_answers",
    "gold",
    "gold_answer",
    "gold_answers",
    "target_answer",
    "target_answers",
    "verification",
    "verifier_result",
    "verification_detail",
    "rubric",
    "rubrics",
    "rubric_items",
    "correct_answer",
    "correct_answers",
    "ground_truth",
    "solution",
    "solutions",
    "hidden_goal",
    "goal_contract",
    "goal_id",
    "target_object",
    "target_receptacle",
    "thought",
    "thoughts",
    "reasoning",
    "analysis",
    "raw_response",
}


def _public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if str(key).casefold() not in _PRIVATE_KEYS
            and not any(
                part in str(key).casefold() for part in ("private", "canary", "gold_", "rubric")
            )
        }
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    if isinstance(value, str):
        # Also drop an unfinished reasoning block; never expose the prefix of it.
        value = re.sub(
            r"<(think|analysis|reasoning)\b[^>]*>.*?(?:</\1\s*>|$)", "", value, flags=re.I | re.S
        )
        value = re.sub(r"</?(?:think|analysis|reasoning)\b[^>]*>", "", value, flags=re.I)
        if value.lstrip().startswith(("{", "[")):
            try:
                return _public_value(json.loads(value))
            except ValueError:
                return "[unparseable structured content omitted]"
    return value


def public_text(value: Any, limit: int = 240) -> str:
    value = _public_value(value)
    text = (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…[truncated]"


def _object(value: Any) -> dict[str, Any]:
    value = _public_value(value)
    return value if isinstance(value, dict) else {}


def _ids(value: Any) -> list[str]:
    return [public_text(item, 80) for item in value[:12]] if isinstance(value, list) else []


def _sample_indices(size: int, limit: int, important: list[int] = ()) -> list[int]:
    if size <= limit:
        return list(range(size))
    selected = {0, size - 1}
    # Cover early/middle/late important events without allowing repeated errors
    # to consume the entire representation; fill the remainder across time.
    room = max(0, limit // 2 - len(selected))
    if important and room:
        selected.update(
            important[round(i * (len(important) - 1) / max(1, room - 1))] for i in range(room)
        )
    for i in range(limit):
        if len(selected) >= limit:
            break
        selected.add(round(i * (size - 1) / max(1, limit - 1)))
    for i in range(size):
        if len(selected) >= limit:
            break
        selected.add(i)
    return sorted(selected)


def _action(raw: Any) -> dict[str, str]:
    value = _object(raw)
    if not value:
        choice = public_text(raw, 20)
        if choice in {"on", "off"}:
            return {"action": "relation_gate", "choice": choice}
        return {"action": "unparsed", "public_text": public_text(raw, 180)}
    fields = (
        "action",
        "agent_id",
        "target",
        "source",
        "relation",
        "relation_type",
        "runtime_route",
        "role",
        "objective",
        "scope",
        "expected_output",
        "prompt",
        "tools",
        "layer",
        "revision_basis",
        "evidence_agent_ids",
        "expected_version",
    )
    return {key: public_text(value[key], 140) for key in fields if key in value}


def _workers(execution: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = execution.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return []
    executed = execution.get("executed_agents")
    if isinstance(executed, list):
        # ExecutionResult also carries cached artifacts from earlier steps.
        # Show newly executed agents here; reuse remains explicit in the step.
        artifacts = {key: value for key, value in artifacts.items() if key in executed}
    result = []
    for agent_id, artifact in sorted(artifacts.items())[:8]:
        if not isinstance(artifact, dict):
            continue
        item: dict[str, Any] = {"agent_id": public_text(agent_id, 80)}
        # Model summaries are PUBLIC observations, not verifier labels. Do not
        # forward raw_response, answer, hidden reasoning, or final trace output.
        for key in ("summary", "unresolved_issues", "integrity_risks", "tool_summary"):
            if artifact.get(key):
                item[key] = public_text(artifact[key], 240)
        turns = [turn for turn in artifact.get("react_trace", []) if isinstance(turn, dict)]
        if turns:
            selected = _sample_indices(len(turns), 3)
            item["tool_steps"] = [
                {
                    "index": i,
                    "action": public_text(turns[i].get("action", {}), 180),
                    "observation": public_text(turns[i].get("observation", {}), 240),
                }
                for i in selected
            ]
            item["tool_steps_total"] = len(turns)
            item["tool_steps_omitted"] = len(turns) - len(selected)
        result.append(item)
    return result


def _graph(value: Any) -> dict[str, Any]:
    graph = _object(value)
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    relations = [r for r in graph.get("relations", []) if isinstance(r, dict)]
    return {
        "nodes": [
            {
                key: public_text(node[key], 140)
                for key in ("agent_id", "prompt", "layer", "allowed_tools")
                if key in node
            }
            for node in nodes[:8]
        ],
        "relations": [
            {
                key: public_text(relation[key], 80)
                for key in ("source", "target", "relation")
                if key in relation
            }
            for relation in relations[:16]
        ],
        "output_agent": public_text(graph.get("output_agent", ""), 80),
        "nodes_total": len(nodes),
        "relations_total": len(relations),
    }


def public_process_trace(
    trace: Any, *, max_events: int = 32, max_chars: int = 10000
) -> dict[str, Any]:
    """Read ``ExecutionTrace.to_dict`` payloads, with coverage and explicit omissions."""
    if max_events < 2 or max_chars < 3000:
        raise ValueError("PATS process evidence bounds cannot retain representative coverage")
    trace = _object(trace)
    events = trace.get("events", [])
    events = events if isinstance(events, list) else []
    steps = []
    actions: Counter[str] = Counter()
    action_instances: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    feedbacks: Counter[str] = Counter()
    rejected = executed = reused = 0
    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "canvas_step":
            continue
        payload = _object(event.get("payload"))
        action = _action(payload.get("raw_action", ""))
        action_name = action.get("action", "unparsed")
        if action_name not in {item.value for item in ActionType} | {"relation_gate"}:
            action_name = "unparsed"
        relation = _object(payload.get("relation_decision"))
        if action_name == "relation_gate":
            action.update(
                {
                    key: public_text(relation[key], 80)
                    for key in ("source", "target", "relation_type")
                    if key in relation
                }
            )
        actions[action_name] += 1
        raw_fields = _object(payload.get("raw_action"))
        signature = {key: raw_fields[key] for key in action if key in raw_fields} or action
        # Count repetitions before text truncation so two long delegations with
        # the same prefix and different objectives are not reported as a loop.
        action_instances[json.dumps(signature, sort_keys=True, ensure_ascii=False)] += 1
        accepted = payload.get("accepted")
        code = public_text(payload.get("rejection_code", ""), 80)
        if accepted is False:
            rejected += 1
            rejections[code or "unspecified"] += 1
        feedback = public_text(payload.get("feedback", ""), 220)
        if feedback:
            feedbacks[feedback] += 1
        execution = _object(payload.get("execution"))
        executed_ids = _ids(payload.get("executed_agents", execution.get("executed_agents", [])))
        reused_ids = _ids(payload.get("reused_agents", execution.get("reused_agents", [])))
        executed += len(executed_ids)
        reused += len(reused_ids)
        item = {
            "sequence": event["sequence"] if type(event.get("sequence")) is int else len(steps),
            "action": action,
            "accepted": accepted if isinstance(accepted, bool) else None,
            "feedback": feedback,
        }
        for key, value in (
            ("rejection_code", code),
            ("executed_agents", executed_ids),
            ("reused_agents", reused_ids),
            ("workers", _workers(execution)),
        ):
            if value:
                item[key] = value
        steps.append(item)
    summary = {
        "steps": len(steps),
        "rejected_actions": rejected,
        "action_counts": dict(sorted(actions.items())),
        "rejection_counts": dict(sorted(rejections.most_common(16))),
        "other_rejections": sum(count for _, count in rejections.most_common()[16:]),
        "worker_executions": executed,
        "worker_reuses": reused,
        "repeated_actions": [
            {"action": public_text(action, 180), "count": count}
            for action, count in action_instances.most_common(6)
            if count >= 3
        ],
        "repeated_feedback": [
            {"feedback": feedback, "count": count}
            for feedback, count in feedbacks.most_common(4)
            if count >= 3
        ],
    }
    important = [
        i for i, item in enumerate(steps) if item["accepted"] is False or item.get("workers")
    ]
    limit = min(max_events, len(steps))
    result = {
        "schema": "pats_public_process",
        "summary": summary,
        "final_graph": _graph(trace.get("final_graph")),
    }
    if len(json.dumps({**result, "events": steps}, ensure_ascii=False)) > max_chars:
        # Prefer a complete decision sequence with shorter public text before
        # dropping any steps. Rich Worker/tool detail is secondary to coverage.
        for item in steps:
            item["feedback"] = public_text(item["feedback"], 140)
            item["action"] = {key: public_text(value, 100) for key, value in item["action"].items()}
            for worker in item.get("workers", []):
                for key in ("summary", "unresolved_issues", "integrity_risks", "tool_summary"):
                    if key in worker:
                        worker[key] = public_text(worker[key], 120)
                for turn in worker.get("tool_steps", []):
                    turn["observation"] = public_text(turn["observation"], 160)
        result["detail_compacted"] = True
    while True:
        selected = _sample_indices(len(steps), limit, important) if steps else []
        result.update(
            events=[steps[i] for i in selected], omitted_events=len(steps) - len(selected)
        )
        if len(json.dumps(result, ensure_ascii=False)) <= max_chars:
            return result
        if limit > 2:
            limit -= 1
            continue
        # A single pathological action can contain many long fields. Preserve
        # the first/last step and all aggregate counts, with tighter text bounds.
        for item in result["events"]:
            item.pop("workers", None)
            item["action"] = {key: public_text(value, 60) for key, value in item["action"].items()}
            item["feedback"] = public_text(item["feedback"], 100)
        result["final_graph"] = {
            key: result["final_graph"][key]
            for key in ("output_agent", "nodes_total", "relations_total")
        }
        result["detail_compacted"] = True
        if len(json.dumps(result, ensure_ascii=False)) > max_chars:
            for item in result["events"]:
                item["action"] = {
                    key: public_text(value, 40)
                    for key, value in item["action"].items()
                    if key in {"action", "target", "source", "agent_id", "choice"}
                }
                item.pop("executed_agents", None)
                item.pop("reused_agents", None)
            for key in ("repeated_actions", "repeated_feedback"):
                summary[key] = [
                    {
                        field: public_text(value, 80) if isinstance(value, str) else value
                        for field, value in item.items()
                    }
                    for item in summary[key][:1]
                ]
            retained_rejections = dict(rejections.most_common(4))
            summary["rejection_counts"] = dict(sorted(retained_rejections.items()))
            summary["other_rejections"] = sum(rejections.values()) - sum(
                retained_rejections.values()
            )
        return result


def process_contrast(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare descriptive process frequencies, without claiming causal benefit.

    Continuous partial rewards stay separate from full successes and zero scores.
    All summaries were calculated before representative trajectory sampling.
    """
    result: dict[str, Any] = {}
    for name, predicate in (
        ("success", lambda r: r == 1),
        ("partial", lambda r: 0 < r < 1),
        ("failure", lambda r: r == 0),
    ):
        subset = [a for a in attempts if predicate(a["reward"])]
        counts: Counter[str] = Counter()
        rejections: Counter[str] = Counter()
        for attempt in subset:
            summary = attempt["trace"]["summary"]
            counts.update(summary["action_counts"])
            rejections.update(summary["rejection_counts"])
        result[name] = {
            "attempts": len(subset),
            "mean_steps": sum(a["trace"]["summary"]["steps"] for a in subset) / len(subset)
            if subset
            else 0,
            "action_counts": dict(sorted(counts.items())),
            "rejection_counts": dict(sorted(rejections.items())),
            "other_rejections": sum(a["trace"]["summary"]["other_rejections"] for a in subset),
            "attempts_with_repeated_actions": sum(
                bool(a["trace"]["summary"]["repeated_actions"]) for a in subset
            ),
            "attempts_with_repeated_feedback": sum(
                bool(a["trace"]["summary"]["repeated_feedback"]) for a in subset
            ),
        }
    success, failure = result["success"], result["failure"]
    result["success_failure_action_rates"] = (
        [
            {
                "action": action,
                "success_per_attempt": success["action_counts"].get(action, 0)
                / success["attempts"],
                "failure_per_attempt": failure["action_counts"].get(action, 0)
                / failure["attempts"],
            }
            for action in sorted(set(success["action_counts"]) | set(failure["action_counts"]))
        ]
        if success["attempts"] and failure["attempts"]
        else []
    )
    return result


def selected_skill_context(manifest: Any) -> dict[str, Any]:
    """Retain the observed card versions and frozen view identity, not arbitrary metadata."""
    manifest = _object(manifest)
    selected = manifest.get("selected", [])
    return {
        **{
            key: public_text(manifest[key], 180)
            for key in ("snapshot_id", "pats_snapshot_id", "pats_scope", "context_sha256")
            if key in manifest
        },
        "selected": [
            {
                "id": public_text(item["id"], 100),
                "version": item.get("version") if type(item.get("version")) is int else None,
            }
            for item in selected[:32]
            if isinstance(item, dict) and "id" in item
        ]
        if isinstance(selected, list)
        else [],
    }


# ---- R2Orch: condition-aware graph evidence ----

_STRUCTURE_FIELDS = ("nodes", "layers", "directed", "bidirectional")


def graph_structure(graph: Any) -> dict[str, Any]:
    """Public structural summary of one final graph (``MultiAgentGraph.to_dict``)."""
    graph = _object(graph)
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    relations = [r for r in graph.get("relations", []) if isinstance(r, dict)]
    roles = []
    for node in nodes:
        delegation = _object(_object(node.get("metadata")).get("director_delegation"))
        role = delegation.get("role")
        if role:
            roles.append(public_text(role, 60).casefold())
    return {
        "nodes": len(nodes),
        "layers": len({node.get("layer") for node in nodes}),
        "directed": sum(r.get("relation") == "directed" for r in relations),
        "bidirectional": sum(r.get("relation") == "bidirectional" for r in relations),
        "roles": sorted(roles),
    }


def structure_summary(structures: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not structures:
        return None
    roles: Counter[str] = Counter()
    for item in structures:
        roles.update(item["roles"])
    return {
        "graphs": len(structures),
        **{key: sum(item[key] for item in structures) / len(structures) for key in _STRUCTURE_FIELDS},
        "role_counts": dict(sorted(roles.most_common(12))),
    }


def within_condition(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Success graphs against failure graphs under one scaffold condition."""
    return {
        "attempts": len(attempts),
        "reward_mean": sum(a["reward"] for a in attempts) / len(attempts),
        "process_contrast": process_contrast(attempts),
        "success_graphs": structure_summary([a["structure"] for a in attempts if a["reward"] == 1]),
        "failure_graphs": structure_summary([a["structure"] for a in attempts if a["reward"] == 0]),
    }


def condition_contrast(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Same task, with cards against without cards: structural difference of the graphs."""
    on = [a for a in attempts if a["condition"] == "on"]
    off = [a for a in attempts if a["condition"] == "off"]
    if not on or not off:
        return None
    on_graph = structure_summary([a["structure"] for a in on])
    off_graph = structure_summary([a["structure"] for a in off])
    return {
        "on": on_graph,
        "off": off_graph,
        "reward_gap": sum(a["reward"] for a in on) / len(on) - sum(a["reward"] for a in off) / len(off),
        "structure_delta": {key: on_graph[key] - off_graph[key] for key in _STRUCTURE_FIELDS},
        "roles_only_on": sorted(set(on_graph["role_counts"]) - set(off_graph["role_counts"])),
        "roles_only_off": sorted(set(off_graph["role_counts"]) - set(on_graph["role_counts"])),
    }


def _card_keys(items: Any) -> list[tuple[str, int | None]]:
    return [
        (str(item["id"]), item.get("version") if type(item.get("version")) is int else None)
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and "id" in item
    ]


def card_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per card id@version: success with the card injected minus success when it was only the reference."""
    on: dict[tuple[str, int | None], list[float]] = {}
    off: dict[tuple[str, int | None], list[float]] = {}
    for row in rows:
        metadata = row.get("metadata", {})
        r2orch = metadata.get("r2orch", {})
        if r2orch.get("scaffold_condition", "on") == "on":
            keys, target = _card_keys(metadata.get("skill_context", {}).get("selected")), on
        else:
            keys, target = _card_keys(r2orch.get("reference_selected")), off
        for key in keys:
            target.setdefault(key, []).append(float(row["reward"]))
    result = []
    for key in sorted(set(on) | set(off), key=lambda k: (k[0], k[1] or 0)):
        a, b = on.get(key, []), off.get(key, [])
        on_mean = sum(a) / len(a) if a else None
        off_mean = sum(b) / len(b) if b else None
        result.append(
            {
                "id": key[0],
                "version": key[1],
                "on_n": len(a),
                "on_mean": on_mean,
                "off_n": len(b),
                "off_mean": off_mean,
                "gap": on_mean - off_mean if a and b else None,
            }
        )
    return result


def unhelpful_cards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cards that were injected into on-condition siblings that then failed."""
    uses: Counter[tuple[str, int | None]] = Counter()
    failed: Counter[tuple[str, int | None]] = Counter()
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("r2orch", {}).get("scaffold_condition", "on") != "on":
            continue
        for key in _card_keys(metadata.get("skill_context", {}).get("selected")):
            uses[key] += 1
            failed[key] += float(row["reward"]) == 0
    return [
        {"id": key[0], "version": key[1], "failed_uses": failed[key], "uses": uses[key]}
        for key in sorted(failed, key=lambda k: (k[0], k[1] or 0))
        if failed[key]
    ]
