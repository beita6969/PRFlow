"""One authoritative Worker artifact syntax check, with actionable diagnostics."""

import json
import math
import re


def summarize_worker_protocol(*, answer: str, raw_response: str, diagnostics: list) -> dict:
    """Bounded public facts from runtime-owned evidence, never model-authored advice.

    Artifact acceptance is distinct from task correctness and final submission.
    Incomplete records stay unknown; raw exceptions are never copied out.
    """
    records = [item for item in diagnostics if isinstance(item, dict)]
    last = records[-1] if records else {}
    attempts = [
        item for item in records if item.get("stage") in ("finalization_1", "finalization_2")
    ]
    status = "unknown"
    if answer == "WORKER_PROTOCOL_FAILURE":
        status = "failed"
    elif answer != "WORKER_BACKEND_FAILURE":
        if last.get("accepted") is False:
            status = "failed"
        elif last.get("accepted") is True:
            status = (
                "recovered"
                if any(item.get("accepted") is False for item in records[:-1])
                else "valid"
            )
        elif not records and raw_response and check_artifact(raw_response)[0] is not None:
            status = "valid"
    stage = None
    error = None
    if status != "unknown":
        stage = (
            "finalization"
            if last.get("stage") in ("finalization_1", "finalization_2")
            else "worker_output"
        )
    if status == "failed":
        reason = last.get("rejection_reason") if last.get("accepted") is False else None
        reason = reason if isinstance(reason, str) else None
        error = {
            "truncated_final_response": "output_truncated",
            "missing_final_json": "invalid_json",
            "empty_answer": "missing_answer",
            "invalid_field_type": "invalid_field_type",
            "generic_acknowledgement": "generic_acknowledgement",
        }.get(reason, "protocol_failure_unknown")
        detail = last.get("parse_error")
        if (
            reason == "missing_final_json"
            and isinstance(detail, dict)
            and detail.get("error_type") == "missing_answer"
        ):
            error = "missing_answer"
        if last.get("accepted") is False and last.get("finish_reason") in ("length", "MAX_TOKENS"):
            error = "output_truncated"
    exhausted = False if status in {"valid", "recovered"} else None
    if status == "failed" and (
        last.get("local_recovery_exhausted") is True or last.get("stage") == "finalization_2"
    ):
        exhausted = True
    return {
        "status": status,
        "stage": stage,
        "error_code": error,
        "finalization_attempts_used": len(attempts) if records or status == "valid" else None,
        "local_recovery_exhausted": exhausted,
    }


def check_artifact(text: str) -> tuple[dict | None, dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    first, last = cleaned.find("{"), cleaned.rfind("}")
    candidate = cleaned[first : last + 1] if first >= 0 and last > first else cleaned
    try:
        payload = json.loads(candidate)
    except (ValueError, TypeError, RecursionError) as exc:
        issue = {"reason": "missing_final_json", "error_type": type(exc).__name__}
        if isinstance(exc, json.JSONDecodeError):
            issue.update(
                message=exc.msg,
                line=exc.lineno,
                column=exc.colno,
                position=exc.pos,
                excerpt=candidate[max(0, exc.pos - 120) : exc.pos + 120],
            )
        return None, issue
    if not isinstance(payload, dict):
        return None, {"reason": "missing_final_json"}
    if "answer" not in payload:
        return None, {"reason": "missing_final_json", "error_type": "missing_answer"}
    if any(key in payload for key in ("action_call", "action_calls", "tool_call", "tool_calls")):
        return None, {"reason": "missing_final_json", "error_type": "tool_call_after_action_phase"}
    answer = payload["answer"]
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return None, {"reason": "empty_answer"}
    if isinstance(answer, bool):
        return None, {
            "reason": "invalid_field_type",
            "field": "answer",
            "expected": "task artifact, not boolean",
        }
    # Models commonly emit the optional tool summary as a single sentence
    # instead of the documented array.  The artifact contract already treats
    # this field as a string-list downstream, so normalize this harmless
    # representation before validation rather than forcing a second remote
    # generation request during finalization.
    tool_summary = payload.get("tool_summary")
    if isinstance(tool_summary, str):
        payload = dict(payload)
        payload["tool_summary"] = [tool_summary]
    elif isinstance(tool_summary, dict):
        # Some models serialize the no-tool state as metadata rather than the
        # documented string array.  Preserve a real action summary when one
        # exists; otherwise the canonical representation is an empty list.
        actions_used = tool_summary.get("actions_used")
        if tool_summary.get("actions_available") is False and not actions_used:
            payload = dict(payload)
            payload["tool_summary"] = []
        else:
            payload = dict(payload)
            payload["tool_summary"] = [
                json.dumps(tool_summary, ensure_ascii=False, sort_keys=True)
            ]
    normalized = re.sub(r"[.!。！]+$", "", str(answer).strip().casefold())
    if normalized in {
        "acknowledged",
        "got it",
        "noted",
        "ok",
        "okay",
        "sure",
        "thank you",
        "thanks",
        "understood",
    }:
        return None, {"reason": "generic_acknowledgement"}
    for field in ("evidence", "unresolved_issues", "tool_summary"):
        if field in payload and not isinstance(payload[field], list):
            return None, {"reason": "invalid_field_type", "field": field, "expected": "array"}
    if "summary" in payload and not isinstance(payload["summary"], str):
        return None, {"reason": "invalid_field_type", "field": "summary", "expected": "string"}
    if "confidence" in payload:
        value = payload["confidence"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return None, {
                "reason": "invalid_field_type",
                "field": "confidence",
                "expected": "finite number in [0,1]",
            }
    return payload, {}


def check_alfworld_terminal_artifact(
    text: str, *, dataset: str, environment_result: dict | None = None
) -> tuple[dict | None, dict]:
    """Normalize a Boolean report only against a runtime-owned ALFWorld terminal result.

    The caller must supply the current Agent's lifecycle result, never model
    text or a peer claim. This changes artifact syntax only; the environment
    remains the authority for reward and submission. The original text is not
    rewritten, and every other field still passes the ordinary schema check.
    """
    payload, issue = check_artifact(text)
    if payload is not None or dataset != "alfworld" or issue != {
        "reason": "invalid_field_type", "field": "answer", "expected": "task artifact, not boolean"
    }:
        return payload, issue
    if not isinstance(environment_result, dict):
        return payload, issue
    result = environment_result
    score = result.get("official_score")
    won = result.get("won")
    if (
        result.get("adapter") != "alfworld"
        or result.get("done") is not True
        or result.get("environment_completed") is not True
        or not isinstance(won, bool)
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or score != float(won)
        or not math.isfinite(score)
        or result.get("termination_reason") not in {
            "official_success", "official_done_without_success", "episode_step_budget_exhausted"
        }
    ):
        return payload, issue
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    candidate = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    normalized = json.loads(candidate)  # check_artifact already established valid JSON above.
    if normalized["answer"] is not won:
        return payload, issue
    normalized["answer"] = json.dumps(normalized["answer"])
    return check_artifact(json.dumps(normalized, ensure_ascii=False))
