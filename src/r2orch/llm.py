from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .action_protocol import ActionCall, ActionSpec
from .backend_failures import (
    BackendRequestError,
    classify_backend_failure,
)
from .config import ModelGatewayConfig
from .deadline import RolloutDeadline
from .director_timeline import TIMELINE_CONTEXT_MODES, director_context_mode
from .qwen_compat import (
    qwen_request_extra,
    recover_qwen_policy_parts,
    response_content,
    response_policy_parts,
)

# Model endpoints are reached directly: inherited HTTP(S)_PROXY settings are ignored.
_DIRECT_OPENER = build_opener(ProxyHandler({}))


def urlopen(request, *, timeout):
    return _DIRECT_OPENER.open(request, timeout=timeout)


_ENDPOINT_FAILOVER_ACTIVE = ContextVar("endpoint_failover_active", default=False)
_ENDPOINT_REQUEST_END = ContextVar("endpoint_request_end", default=None)
_ENDPOINT_QUEUE_WAIT_CAP_S = ContextVar("endpoint_queue_wait_cap_s", default=None)


@contextmanager
def endpoint_failover_scope(end, *, queue_wait_cap_s: float | None = None):
    active = _ENDPOINT_FAILOVER_ACTIVE.set(True)
    budget = _ENDPOINT_REQUEST_END.set(end)
    queue_cap = _ENDPOINT_QUEUE_WAIT_CAP_S.set(queue_wait_cap_s)
    try:
        yield
    finally:
        _ENDPOINT_QUEUE_WAIT_CAP_S.reset(queue_cap)
        _ENDPOINT_REQUEST_END.reset(budget)
        _ENDPOINT_FAILOVER_ACTIVE.reset(active)


_FINALIZATION_REQUEST = ContextVar("worker_finalization_request", default=False)
_DIRECTOR_RECOVERY = ContextVar("director_bounded_recovery", default=False)


@contextmanager
def director_recovery_budget(enabled: bool):
    token = _DIRECTOR_RECOVERY.set(enabled)
    try:
        yield
    finally:
        _DIRECTOR_RECOVERY.reset(token)


@dataclass
class RequestTokenCredit:
    limit: int
    token_in: int = 0
    token_out: int = 0
    pre_reserved_closure_tokens: int | None = None
    cap_output: bool = False


class RequestTokenCreditExceeded(RuntimeError):
    def __init__(
        self, required: int, credit: RequestTokenCredit, budget: dict[str, Any] | None = None
    ):
        super().__init__("insufficient per-request Worker token credit")
        self.required = required
        self.credit = credit
        self.budget = dict(budget or {})
        self.request_events = list(_current_request_events())


_TOKEN_CREDIT: ContextVar[RequestTokenCredit | None] = ContextVar(
    "worker_token_credit", default=None
)


@contextmanager
def request_token_credit(
    limit: int | None, *, pre_reserved_closure_tokens: int | None = None, cap_output: bool = False
):
    credit = (
        RequestTokenCredit(
            max(0, limit),
            pre_reserved_closure_tokens=pre_reserved_closure_tokens,
            cap_output=cap_output,
        )
        if limit is not None
        else None
    )
    token = _TOKEN_CREDIT.set(credit)
    try:
        yield credit
    finally:
        _TOKEN_CREDIT.reset(token)


BUDGET_SEMANTICS = "utf8_request_upper_bound"


def request_budget_quote(request: Mapping[str, Any]) -> dict[str, Any]:
    """The same conservative quote used immediately before an OpenAI send."""
    body = {key: request[key] for key in ("messages", "tools") if key in request}
    input_bound = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 2048
    output_bound = int(request.get("max_completion_tokens", request.get("max_tokens", 2048)))
    if output_bound < 0:
        raise ValueError("negative request output budget")
    return {
        "semantics": BUDGET_SEMANTICS,
        "input_bound": input_bound,
        "output_bound": output_bound,
        "required_tokens": input_bound + output_bound,
    }


def request_admission(
    quote: Mapping[str, Any],
    *,
    credit: int,
    spent: int = 0,
    pre_reserved_closure_tokens: int | None = None,
) -> dict[str, Any]:
    # Canvas already withheld the baseline reserve. Protect only the additional
    # amount revealed by the ACTUAL request; never subtract that baseline twice.
    extra = (
        max(0, 2 * int(quote["required_tokens"]) - pre_reserved_closure_tokens)
        if pre_reserved_closure_tokens is not None
        else 0
    )
    required = int(quote["required_tokens"]) + extra
    return {
        **quote,
        "credit": credit,
        "spent": spent,
        "pre_reserved_closure_tokens": pre_reserved_closure_tokens,
        "additional_closure_reserve": extra,
        "required_with_reserve": required,
        "admitted": spent + required <= credit,
    }


def _request_credit_admission(
    request: dict[str, Any], credit: RequestTokenCredit
) -> dict[str, Any]:
    """QA may shrink completion allowance, never input/history or the hard limit."""
    key = "max_tokens"
    configured = int(request.get(key, 2048))
    quote = request_budget_quote(request)
    if credit.cap_output:
        available = credit.limit - credit.token_in - credit.token_out - quote["input_bound"]
        admitted_cap = min(configured, max(0, available))
        if admitted_cap < min(configured, 128) or admitted_cap <= 0:
            budget = request_admission(
                quote, credit=credit.limit, spent=credit.token_in + credit.token_out
            )
            budget.update(admitted=False, stop_reason="insufficient_input_and_output_credit")
            raise RequestTokenCreditExceeded(quote["required_tokens"], credit, budget)
        request[key] = admitted_cap
        quote = {
            **quote,
            "output_bound": admitted_cap,
            "required_tokens": quote["input_bound"] + admitted_cap,
            "configured_output_bound": configured,
            "output_cap_reduced": admitted_cap < configured,
        }
    budget = request_admission(
        quote,
        credit=credit.limit,
        spent=credit.token_in + credit.token_out,
        pre_reserved_closure_tokens=credit.pre_reserved_closure_tokens,
    )
    if not budget["admitted"]:
        raise RequestTokenCreditExceeded(budget["required_tokens"], credit, budget)
    return budget


@contextmanager
def worker_finalization_request():
    """Runtime owns content retries in this scope; transport retry policy is unchanged."""
    token = _FINALIZATION_REQUEST.set(True)
    try:
        yield
    finally:
        _FINALIZATION_REQUEST.reset(token)


class DirectorContextExhausted(ValueError):
    """Exact policy history cannot fit another action; history is never truncated."""


def director_static_budget(prompt_tokens: int, requested: int) -> int:
    if prompt_tokens < 0 or requested <= 0:
        raise ValueError("invalid Director token budget")
    available = 32768 - prompt_tokens - 64
    if available < min(requested, 128):
        raise DirectorContextExhausted("Director context exhausted; exact history preserved")
    return min(requested, available)


class _PriorityRequestGate:
    """Endpoint-wide bounded gate that lets primary calls pass queued probes."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self._condition = threading.Condition()
        self._in_flight = 0
        self._primary_waiters = 0

    def acquire(self, *, priority: str, timeout: float | None = None) -> bool:
        primary = priority != "counterfactual"
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            if primary:
                self._primary_waiters += 1
            try:
                while self._in_flight >= self.limit or (not primary and self._primary_waiters > 0):
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
                self._in_flight += 1
                return True
            finally:
                if primary:
                    self._primary_waiters -= 1

    def release(self) -> None:
        with self._condition:
            if self._in_flight <= 0:
                raise RuntimeError("request gate released without an acquisition")
            self._in_flight -= 1
            self._condition.notify_all()


_REQUEST_GATE_LOCK = threading.Lock()
_REQUEST_GATES: dict[tuple[str, str, int], _PriorityRequestGate] = {}
_REQUEST_PRIORITY = threading.local()
_REQUEST_EVENT_TARGET = threading.local()


@dataclass(frozen=True)
class _RequestSlotInfo:
    route: str
    priority: str
    request_budget_s: float
    timeout_s: float
    queue_wait_s: float
    request_started_monotonic: float


@contextmanager
def _capture_request_events(target: list[dict[str, Any]], *, role: str = ""):
    previous = getattr(_REQUEST_EVENT_TARGET, "value", None)
    previous_role = getattr(_REQUEST_EVENT_TARGET, "role", "")
    _REQUEST_EVENT_TARGET.value = target
    _REQUEST_EVENT_TARGET.role = role
    try:
        yield
    finally:
        _REQUEST_EVENT_TARGET.value = previous
        _REQUEST_EVENT_TARGET.role = previous_role


def _emit_request_event(event: dict[str, Any]) -> None:
    import uuid

    target = getattr(_REQUEST_EVENT_TARGET, "value", None)
    if isinstance(target, list):
        target.append(
            {
                **event,
                "event_id": uuid.uuid4().hex,
                "request_role": getattr(_REQUEST_EVENT_TARGET, "role", ""),
            }
        )


def _attach_provider_usage(events, usage, *, surface, model, effort=None):
    """Attach exact provider counters; missing fields stay None, never estimated zero."""

    def get(obj, key):
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    inp, out = ("prompt_tokens", "completion_tokens")
    fields = {
        "input_tokens": get(usage, inp),
        "output_tokens": get(usage, out),
        "reasoning_tokens": get(get(usage, out + "_details"), "reasoning_tokens"),
        "cached_tokens": get(get(usage, inp + "_details"), "cached_tokens"),
    }
    for event in reversed(events):
        if event.get("event") == "backend_request_success":
            event.update(
                provider_usage={
                    key: value if isinstance(value, (int, float)) and math.isfinite(value) else None
                    for key, value in fields.items()
                },
                usage_source="provider_reported",
                api_surface=surface,
                provider_model=model,
                reasoning_effort=effort,
            )
            break


def _current_request_events() -> list[dict[str, Any]]:
    target = getattr(_REQUEST_EVENT_TARGET, "value", None)
    if not isinstance(target, list):
        return []
    return [dict(event) for event in target if isinstance(event, dict)]


def _request_success_event(
    slot: _RequestSlotInfo,
    *,
    upstream_elapsed_s: float,
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "backend_request_success",
        "stage": "backend_request",
        "route": slot.route,
        "priority": slot.priority,
        "attempt": int(attempt),
        "request_budget_s": slot.request_budget_s,
        "queue_wait_s": slot.queue_wait_s,
        "upstream_elapsed_s": float(upstream_elapsed_s),
        "total_elapsed_s": time.monotonic() - slot.request_started_monotonic,
    }


def _request_failure_event(
    classification: Any,
    *,
    priority: str,
    attempt: int,
    will_retry: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "backend_request_failure",
        **classification.to_dict(),
        "priority": priority,
        "attempt": int(attempt),
        "will_retry": bool(will_retry),
    }


def _request_gate(config: ModelGatewayConfig) -> _PriorityRequestGate:
    """Share one bounded in-flight gate across clients for the same endpoint."""

    limit = int(config.max_concurrency)
    if limit <= 0:
        raise ValueError("model gateway max_concurrency must be positive")
    key = (
        config.base_url.rstrip("/"),
        hashlib.sha256(config.api_key.encode()).hexdigest(),
        limit,
    )
    with _REQUEST_GATE_LOCK:
        gate = _REQUEST_GATES.get(key)
        if gate is None:
            gate = _PriorityRequestGate(limit)
            _REQUEST_GATES[key] = gate
        return gate


def current_request_priority() -> str:
    return str(getattr(_REQUEST_PRIORITY, "value", "primary"))


@contextmanager
def request_priority(priority: str):
    """Mark requests in the current thread as primary or counterfactual."""

    if priority not in {"primary", "counterfactual"}:
        raise ValueError("request priority must be primary or counterfactual")
    previous = getattr(_REQUEST_PRIORITY, "value", "primary")
    _REQUEST_PRIORITY.value = priority
    try:
        yield
    finally:
        _REQUEST_PRIORITY.value = previous


def _credit_failed_request(deadline, exc, started):
    if deadline is not None and hasattr(deadline, "record_failed_request"):
        failure = classify_backend_failure(exc)
        if failure.backend_failure:
            deadline.record_failed_request(started, time.monotonic())


@contextmanager
def _failed_request_clock(deadline):
    started = time.monotonic()
    try:
        yield
    except Exception as exc:
        _credit_failed_request(deadline, exc, started)
        raise


@contextmanager
def _request_slot(
    config: ModelGatewayConfig,
    deadline: RolloutDeadline | None,
    *,
    request_budget_cap_s: float | None = None,
    attempt: int = 1,
):
    gate = _request_gate(config)
    priority = str(getattr(_REQUEST_PRIORITY, "value", "primary"))
    pause = (
        deadline.pause_no_progress("backend_request")
        if deadline is not None and hasattr(deadline, "pause_no_progress")
        else nullcontext()
    )
    with pause, _failed_request_clock(deadline):
        request_started = time.monotonic()
        if deadline is None:
            request_budget_s = config.timeout_s
        else:
            request_budget_s = deadline.request_budget_s(
                "backend_queue", route=config.route_name
            )
            request_budget_s = min(config.timeout_s, request_budget_s)
        if request_budget_cap_s is not None:
            request_budget_s = min(request_budget_s, request_budget_cap_s)
        request_budget_s = max(0.001, request_budget_s)
        queue_wait_s = request_budget_s
        queue_wait_cap_s = _ENDPOINT_QUEUE_WAIT_CAP_S.get()
        if queue_wait_cap_s is not None:
            queue_wait_s = min(queue_wait_s, max(0.0, float(queue_wait_cap_s)))
        acquired = gate.acquire(priority=priority, timeout=queue_wait_s)
        if not acquired:
            _credit_failed_request(deadline, TimeoutError("backend queue timeout"), request_started)
            if deadline is not None:
                deadline.check("backend_queue")
            queue_elapsed_s = time.monotonic() - request_started
            classification = classify_backend_failure(
                TimeoutError(f"backend queue wait exceeded {queue_wait_s:.1f} seconds"),
                stage="backend_queue",
                route=config.route_name,
                queue_wait_s=queue_elapsed_s,
                total_elapsed_s=queue_elapsed_s,
                request_budget_s=request_budget_s,
            )
            _emit_request_event(
                _request_failure_event(
                    classification,
                    priority=priority,
                    attempt=attempt,
                    will_retry=False,
                )
            )
            raise BackendRequestError(classification, request_events=_current_request_events())
        try:
            queue_elapsed_s = time.monotonic() - request_started
            if deadline is None:
                timeout_s = request_budget_s - queue_elapsed_s
            else:
                remaining_request_s = request_budget_s - queue_elapsed_s
                hard_remaining = getattr(deadline, "hard_remaining_s", None)
                timeout_s = min(
                    remaining_request_s,
                    (
                        hard_remaining("backend_request")
                        if callable(hard_remaining)
                        else remaining_request_s
                    ),
                )
            if timeout_s <= 0:
                classification = classify_backend_failure(
                    TimeoutError("backend queue consumed the complete request wall-clock budget"),
                    stage="backend_queue",
                    route=config.route_name,
                    queue_wait_s=queue_elapsed_s,
                    total_elapsed_s=time.monotonic() - request_started,
                    request_budget_s=request_budget_s,
                )
                _emit_request_event(
                    _request_failure_event(
                        classification,
                        priority=priority,
                        attempt=attempt,
                        will_retry=False,
                    )
                )
                raise BackendRequestError(classification, request_events=_current_request_events())
            yield _RequestSlotInfo(
                route=config.route_name,
                priority=priority,
                request_budget_s=request_budget_s,
                timeout_s=max(0.001, timeout_s),
                queue_wait_s=queue_elapsed_s,
                request_started_monotonic=request_started,
            )
        finally:
            gate.release()


def _logical_request_budget_s(
    config: ModelGatewayConfig,
    deadline: RolloutDeadline | None,
) -> float:
    if deadline is None:
        budget = config.timeout_s
    else:
        try:
            budget = deadline.request_budget_s("backend_queue", route=config.route_name)
        except TypeError:
            budget = deadline.request_budget_s("backend_queue")
        budget = min(config.timeout_s, budget)
    endpoint_end = _ENDPOINT_REQUEST_END.get()
    if endpoint_end is not None:
        budget = min(budget, max(0.001, endpoint_end - time.monotonic()))
    return budget


RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_BACKOFF_BASE_S = 2.0
RATE_LIMIT_BACKOFF_CAP_S = 30.0


def _mark_last_failure_for_retry(*, delay_s: float = 0.0, retry_limit: int = 1) -> None:
    target = getattr(_REQUEST_EVENT_TARGET, "value", None)
    if isinstance(target, list) and target:
        target[-1]["will_retry"] = True
        target[-1]["retry_delay_s"] = delay_s
        target[-1]["retry_limit"] = retry_limit


def _retry_backend_request(
    error: BackendRequestError,
    *,
    attempt: int,
    sequence_started: float,
    sequence_budget_s: float,
    deadline: RolloutDeadline | None = None,
) -> bool:
    rate_limited = error.classification.kind == "rate_limit"
    if _ENDPOINT_FAILOVER_ACTIVE.get():
        return False  # The pool owns endpoint failover and whole-pool retry.
    retry_limit = RATE_LIMIT_MAX_RETRIES if rate_limited else 1
    if attempt > retry_limit or not error.classification.retryable:
        return False
    backoff_s = (
        min(RATE_LIMIT_BACKOFF_CAP_S, RATE_LIMIT_BACKOFF_BASE_S * 2 ** (attempt - 1))
        if rate_limited
        else 0.25
    )
    # Never retry earlier than the provider permits. A zero Retry-After still
    # gets a local backoff so concurrent clients cannot busy-loop on capacity.
    delay_s = max(backoff_s, error.classification.retry_after_s or 0.0)
    remaining_s = sequence_budget_s - (time.monotonic() - sequence_started)
    if remaining_s <= delay_s + 0.001:
        return False
    _mark_last_failure_for_retry(delay_s=delay_s, retry_limit=retry_limit)
    pause = (
        deadline.pause_no_progress("backend_retry_wait") if deadline is not None else nullcontext()
    )
    with pause:
        remaining_delay = delay_s
        while remaining_delay > 0:
            if deadline is not None:
                deadline.check("backend_retry_wait")
            interval = min(1.0, remaining_delay)
            time.sleep(interval)
            remaining_delay -= interval
        if deadline is not None:
            deadline.check("backend_retry_wait_complete")
    return True


_VLLM_CONTEXT_OVERFLOW = re.compile(
    r"This model's maximum context length is (\d+) tokens\. However, you requested "
    r"(\d+) output tokens and your prompt contains (?:at least )?(\d+) input tokens, "
    r"for a total of (?:at least )?(\d+) tokens\."
)


def _correct_qwen_worker_context_budget(
    error: BackendRequestError,
    config: ModelGatewayConfig,
    request: dict[str, Any],
    deadline: RolloutDeadline | None,
    *,
    sequence_started: float,
    sequence_budget_s: float,
) -> bool:
    """Count the unchanged chat on its server after an explicit context rejection."""

    if (
        getattr(config, "request_profile", "") != "qwen"
        or getattr(_REQUEST_EVENT_TARGET, "role", "") != "worker"
        or error.classification.status_code != 400
        or error.classification.kind != "invalid_request"
    ):
        return False
    match = _VLLM_CONTEXT_OVERFLOW.search(error.classification.message)
    if match is None:
        return False
    context_limit, rejected_output, prompt_lower_bound, rejected_total = map(int, match.groups())
    token_key = "max_tokens"
    original_cap = request.get(token_key)
    if (
        type(original_cap) is not int
        or original_cap != rejected_output
        or original_cap <= 0
        or rejected_total != prompt_lower_bound + rejected_output
        or rejected_total <= context_limit
    ):
        return False
    remaining_s = sequence_budget_s - (time.monotonic() - sequence_started)
    if remaining_s <= 0.001:
        return False
    extra = request.get("extra_body", {})
    body = {"model": request["model"], "messages": request["messages"],
            "add_generation_prompt": True}
    # Match the actual chat renderer, including native tools and extra-body
    # template overrides. /tokenize itself has no output cap or early cutoff.
    for key in (
        "tools", "chat_template", "chat_template_kwargs", "add_generation_prompt",
        "continue_final_message", "add_special_tokens", "media_io_kwargs", "mm_processor_kwargs",
    ):
        if key in request or key in extra:
            body[key] = extra.get(key, request.get(key))
    counted_request = Request(
        config.base_url.rstrip("/").removesuffix("/v1") + "/tokenize",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + config.api_key, "Content-Type": "application/json"},
    )
    pause = deadline.pause_no_progress("backend_tokenize") if deadline is not None else nullcontext()
    count_started = time.monotonic()
    try:
        with pause, urlopen(counted_request, timeout=min(30.0, remaining_s)) as counted:
            payload = json.load(counted)
        prompt_count = payload["count"]
        if type(prompt_count) is not int or prompt_count < prompt_lower_bound:
            return False
        if payload.get("max_model_len", context_limit) != context_limit:
            return False
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError):
        return False  # Retain the original rejection; do not guess or truncate input.
    corrected_cap = context_limit - prompt_count
    if not 0 < corrected_cap < original_cap:
        return False
    if sequence_budget_s - (time.monotonic() - sequence_started) <= 0.001:
        return False
    if deadline is not None:
        deadline.check("backend_context_correction")
    request[token_key] = corrected_cap
    _mark_last_failure_for_retry(retry_limit=1)
    _emit_request_event({
        "schema_version": 1, "event": "backend_context_budget_correction",
        "stage": "backend_tokenize", "route": config.route_name,
        "context_limit": context_limit, "reported_prompt_lower_bound": prompt_lower_bound,
        "prompt_tokens": prompt_count, "count_source": "same_endpoint_tokenize",
        "previous_max_output_tokens": original_cap, "max_output_tokens": corrected_cap,
        "tokenize_elapsed_s": time.monotonic() - count_started,
        "remaining_request_budget_s": sequence_budget_s - (time.monotonic() - sequence_started),
        "retry_limit": 1,
    })
    return True


def _openai_completion_create(
    client: Any,
    config: ModelGatewayConfig,
    request: dict[str, Any],
    deadline: RolloutDeadline | None,
):
    sequence_started = time.monotonic()
    sequence_budget_s = _logical_request_budget_s(config, deadline)
    context_corrected = False
    attempt = 0
    while attempt < RATE_LIMIT_MAX_RETRIES + 1 + int(context_corrected):
        attempt += 1
        remaining_s = max(
            0.001,
            sequence_budget_s - (time.monotonic() - sequence_started),
        )
        try:
            return _openai_completion_attempt(
                client,
                config,
                request,
                deadline,
                attempt=attempt,
                request_budget_cap_s=remaining_s,
            )
        except BackendRequestError as exc:
            if not context_corrected and _correct_qwen_worker_context_budget(
                exc, config, request, deadline,
                sequence_started=sequence_started, sequence_budget_s=sequence_budget_s,
            ):
                context_corrected = True
                continue
            if _retry_backend_request(
                exc,
                attempt=attempt - int(context_corrected),
                sequence_started=sequence_started,
                sequence_budget_s=sequence_budget_s,
                deadline=deadline,
            ):
                continue
            raise
    raise AssertionError("bounded OpenAI completion retry loop did not terminate")


def _openai_completion_attempt(
    client: Any,
    config: ModelGatewayConfig,
    request: dict[str, Any],
    deadline: RolloutDeadline | None,
    *,
    attempt: int,
    request_budget_cap_s: float,
):
    credit = _TOKEN_CREDIT.get()
    budget = None
    if credit is not None:
        # Conservative UTF-8 byte estimate plus chat/schema framing. This checks
        # EACH provider request, including a length repair, before it is billed.
        budget = _request_credit_admission(request, credit)
    # Separate OpenAI requests are intentionally kept separate.  Concurrent
    # callers become a continuous server-side batch in vLLM/SGLang while this
    # endpoint-level gate protects fixed remote runtimes from overload.
    with _request_slot(
        config,
        deadline,
        request_budget_cap_s=request_budget_cap_s,
        attempt=attempt,
    ) as slot:
        upstream_started = time.monotonic()
        try:
            response = client.chat.completions.create(**request, timeout=slot.timeout_s)
        except Exception as exc:
            # Provider clients expose several timeout exception classes.  If the
            # request consumed the shared hard/no-progress budget, normalize it
            # to the rollout deadline error so the caller persists a timeout
            # snapshot and applies the bounded replacement policy.  Immediate
            # provider errors still pass through unchanged.
            _credit_failed_request(deadline, exc, slot.request_started_monotonic)
            if deadline is not None:
                deadline.check("backend_request_failed")
            upstream_elapsed_s = time.monotonic() - upstream_started
            classification = classify_backend_failure(
                exc,
                stage="backend_request",
                route=config.route_name,
                queue_wait_s=slot.queue_wait_s,
                upstream_elapsed_s=upstream_elapsed_s,
                total_elapsed_s=time.monotonic() - slot.request_started_monotonic,
                request_budget_s=slot.request_budget_s,
            )
            failure_event = _request_failure_event(
                classification, priority=slot.priority, attempt=attempt, will_retry=False,
            )
            _emit_request_event(failure_event)
            if classification.backend_failure:
                raise BackendRequestError(
                    classification, request_events=_current_request_events()
                ) from exc
            raise
        event = _request_success_event(
            slot,
            upstream_elapsed_s=time.monotonic() - upstream_started,
            attempt=attempt,
        )
        # Persist known usage immediately, even if a later length-repair request fails.
        response_usage = getattr(response, "usage", None)
        event["completion_usage"] = {
            "token_in": int(getattr(response_usage, "prompt_tokens", 0) or 0),
            "token_out": int(getattr(response_usage, "completion_tokens", 0) or 0),
        }
        if budget is not None:
            event["request_token_budget"] = budget
        if credit is not None:
            credit.token_in += event["completion_usage"]["token_in"]
            credit.token_out += event["completion_usage"]["token_out"]
        _emit_request_event(event)
        return response


@dataclass
class LLMResponse:
    text: str
    model: str
    action_calls: list[ActionCall] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    token_in: int = 0
    token_out: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_reasoning_text: str = ""
    raw_action_text: str = ""
    prompt_token_ids: tuple[int, ...] = ()
    completion_token_ids: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] = ()
    token_provenance: str = "unavailable"
    training_eligible: bool = False


class BinaryChoiceUnavailable(RuntimeError):
    """The route cannot expose an auditable constrained binary policy."""


@dataclass(frozen=True)
class BinaryChoiceResponse:
    """One sampled off/on token plus the post-mask behavior probabilities."""

    choice: str
    model: str
    probabilities: dict[str, float]
    log_probabilities: dict[str, float]
    token_ids: dict[str, int]
    prompt_token_ids: tuple[int, ...] = ()
    token_in: int = 0
    token_out: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.choice not in {"off", "on"}:
            raise ValueError("binary choice must be exactly off or on")
        if set(self.probabilities) != {"off", "on"}:
            raise ValueError("binary probabilities must contain exactly off and on")
        if set(self.log_probabilities) != {"off", "on"}:
            raise ValueError("binary log probabilities must contain exactly off and on")
        if set(self.token_ids) != {"off", "on"}:
            raise ValueError("binary token IDs must contain exactly off and on")
        if self.token_ids["off"] == self.token_ids["on"]:
            raise ValueError("off and on must have distinct tokenizer IDs")
        values = tuple(float(self.probabilities[key]) for key in ("off", "on"))
        logs = tuple(float(self.log_probabilities[key]) for key in ("off", "on"))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("binary probabilities must be finite values in [0, 1]")
        if not all(math.isfinite(value) for value in logs):
            raise ValueError("binary log probabilities must be finite")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("binary probabilities must sum to one")

    def to_policy_audit(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "log_probabilities": dict(self.log_probabilities),
            "token_ids": dict(self.token_ids),
            "model": self.model,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "metadata": dict(self.metadata),
        }


class LabelChoiceUnavailable(RuntimeError):
    """The route cannot expose an auditable constrained single-label policy."""


@dataclass(frozen=True)
class LabelChoiceResponse:
    """One sampled candidate label plus its post-mask behavior log-probabilities."""

    label: str
    log_probabilities: dict[str, float]
    token_ids: dict[str, int]
    prompt_token_ids: tuple[int, ...] = ()
    model: str = ""
    token_in: int = 0
    token_out: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.log_probabilities) != set(self.token_ids) or self.label not in self.token_ids:
            raise ValueError("label choice must cover exactly the constrained label support")
        if not all(math.isfinite(float(value)) for value in self.log_probabilities.values()):
            raise ValueError("label log probabilities must be finite")


class ChatBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        actions: Sequence[ActionSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMResponse: ...

    def choose_binary(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        token_ids: dict[str, int],
    ) -> BinaryChoiceResponse: ...


class MockBackend:
    """Deterministic backend following MANTA's offline-fallback testing style."""

    def __init__(
        self,
        responses: Iterable[str | LLMResponse] | None = None,
        *,
        handler: Callable[[list[dict[str, Any]], str], str | LLMResponse] | None = None,
        binary_responses: Iterable[BinaryChoiceResponse] | None = None,
        label_responses: Iterable[LabelChoiceResponse] | None = None,
    ) -> None:
        self.responses = deque(responses or [])
        self.handler = handler
        self.binary_responses = deque(binary_responses or [])
        self.label_responses = deque(label_responses or [])
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        actions: Sequence[ActionSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "role": role,
                "actions": [action.to_context_dict() for action in actions],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
            }
        )
        if self.handler is not None:
            result = self.handler(messages, role)
        elif self.responses:
            result = self.responses.popleft()
        else:
            result = json.dumps(
                {
                    "answer": f"mock answer from {role}",
                    "summary": f"mock summary from {role}",
                    "confidence": 0.5,
                    "evidence": [],
                    "unresolved_issues": [],
                }
            )
        if isinstance(result, LLMResponse):
            return result
        text = str(result)
        approx_in = sum(len(str(message.get("content", ""))) for message in messages) // 4
        return LLMResponse(
            text=text,
            model=f"mock:{role}",
            token_in=approx_in,
            token_out=max(1, len(text) // 4),
            metadata={"mock": True},
            raw_action_text=text,
            token_provenance="mock_text",
            training_eligible=True,
        )

    def choose_binary(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        token_ids: dict[str, int],
    ) -> BinaryChoiceResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "role": role,
                "binary_choices": ["off", "on"],
                "token_ids": dict(token_ids),
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 1,
                "enable_thinking": False,
            }
        )
        if not self.binary_responses:
            raise BinaryChoiceUnavailable(
                "MockBackend requires an explicit BinaryChoiceResponse; no 0.5 fallback is allowed"
            )
        response = self.binary_responses.popleft()
        if dict(response.token_ids) != dict(token_ids):
            raise BinaryChoiceUnavailable("mock binary response tokenizer attestation mismatch")
        return response

    def choose_label(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        token_ids: dict[str, int],
        seed: int,
        temperature: float = 1.0,
    ) -> LabelChoiceResponse:
        """Explicit queued responses first; otherwise a uniform, seed-determined label."""

        self.calls.append(
            {
                "messages": list(messages),
                "role": role,
                "label_choices": list(token_ids),
                "token_ids": dict(token_ids),
                "seed": int(seed),
                "temperature": temperature,
                "top_p": 1.0,
                "max_tokens": 1,
                "enable_thinking": False,
            }
        )
        if self.label_responses:
            return self.label_responses.popleft()
        labels = list(token_ids)
        digest = hashlib.sha256(f"{int(seed)}\0{role}".encode()).digest()
        label = labels[int.from_bytes(digest[:8], "big") % len(labels)]
        uniform = -math.log(len(labels))
        return LabelChoiceResponse(
            label=label,
            log_probabilities={item: uniform for item in labels},
            token_ids=dict(token_ids),
            model=f"mock:{role}",
            metadata={"mock": True, "sampling_seed": int(seed)},
        )


class OpenAICompatibleBackend:
    """Role-isolated OpenAI-compatible gateway for the local Qwen3.5-9B service."""

    def __init__(self, config: ModelGatewayConfig | None = None) -> None:
        self.config = config or ModelGatewayConfig()
        self.rollout_deadline: RolloutDeadline | None = None
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAICompatibleBackend requires the 'openai' extra: pip install -e '.[openai]'"
            ) from exc

        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_s,
            # SDK retries are disabled. The gateway owns bounded typed retries
            # inside the original logical request budget and records every attempt.
            max_retries=0,
            **self._proxy_client_options(),
        )

    def _proxy_client_options(self) -> dict[str, Any]:
        from openai import DefaultHttpxClient

        return {"http_client": DefaultHttpxClient(proxy=None, trust_env=False)}

    def set_deadline_context(self, deadline: RolloutDeadline | None) -> None:
        self.rollout_deadline = deadline

    def close(self) -> None:
        self.client.close()

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        schema: dict[str, Any],
    ) -> LLMResponse:
        """Constrain Skill Refiner output through the ordinary request gateway."""

        if role != "skill-distiller":
            raise ValueError("JSON schema responses are limited to skill-distiller")
        if not isinstance(schema, dict):
            raise TypeError("response JSON schema must be an object")
        return self.generate(messages, role=role, response_json_schema=schema)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        actions: Sequence[ActionSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        response_json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if response_json_schema is not None and (role != "skill-distiller" or actions):
            raise ValueError("JSON schema responses require skill-distiller without tool actions")
        deadline = getattr(self, "rollout_deadline", None)
        request_events: list[dict[str, Any]] = []
        if deadline is not None:
            deadline.check(f"{role}_request_start")
            scope_request = getattr(deadline, "for_request", None)
            if callable(scope_request):
                deadline = scope_request(role, self.config.timeout_s)
        role_config = self.config.roles.get(role, self.config.roles.get("worker"))
        if role_config is None:
            raise KeyError(f"model role is not configured: {role}")
        thinking_enabled = (
            role_config.enable_thinking if enable_thinking is None else bool(enable_thinking)
        )
        qwen_text_actions = bool(actions and self.config.request_profile == "qwen")
        if qwen_text_actions:
            converted = _text_action_messages(messages, actions)
            # Qwen's chat template permits one leading system message.
            system_parts = [str(message.get("content") or "") for message in converted
                            if message.get("role") in {"system", "developer"}]
            request_messages = [
                {"role": "system", "content": "\n\n".join(system_parts)},
                *[message for message in converted if message.get("role") not in {"system", "developer"}],
            ]
        else:
            request_messages = _openai_messages(messages)
        if role_config.system_prompt:
            if qwen_text_actions:
                request_messages[0]["content"] = role_config.system_prompt + "\n\n" + request_messages[0]["content"]
            else:
                request_messages = [
                    {"role": "system", "content": role_config.system_prompt},
                    *request_messages,
                ]
        requested_temperature = role_config.temperature if temperature is None else temperature
        sent_temperature = requested_temperature
        requested_max_tokens = role_config.max_tokens if max_tokens is None else max_tokens
        request: dict[str, Any] = {
            "model": role_config.model,
            "messages": request_messages,
            "temperature": sent_temperature,
            "top_p": role_config.top_p,
        }
        if response_json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "pats_skill_review",
                    "strict": True,
                    "schema": response_json_schema,
                },
            }
        if self.config.sampling_seed is not None:
            request["seed"] = int(self.config.sampling_seed)
        exact_policy_trace_requested = bool(
            role in {"graph-director", "proposer"} and self.config.request_profile == "qwen"
        )
        if exact_policy_trace_requested:
            # The sampled-token probabilities are part of the behavior-policy
            # record, not optional trainer diagnostics reconstructed later.
            request.update(logprobs=True, top_logprobs=0)
        request["max_tokens"] = requested_max_tokens
        if role_config.reasoning_effort is not None:
            request["reasoning_effort"] = role_config.reasoning_effort
        if actions and not qwen_text_actions:
            tools = [_openai_tool(action) for action in actions]
            request.update(tools=tools, tool_choice="auto")
            request["parallel_tool_calls"] = False
        if self.config.request_profile == "qwen":
            request["extra_body"] = {
                **qwen_request_extra(
                    enable_thinking=thinking_enabled,
                    director_timeline=(
                        role == "graph-director"
                        and director_context_mode() in TIMELINE_CONTEXT_MODES
                    ),
                ),
                **({"top_k": role_config.top_k} if role_config.top_k is not None else {}),
                **(
                    {"return_token_ids": True, "skip_special_tokens": False}
                    if exact_policy_trace_requested
                    else {}
                ),
            }
        if role == "graph-director" and self.config.request_profile == "qwen":
            body = {
                "model": role_config.model,
                "messages": request_messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": request["extra_body"]["chat_template_kwargs"],
            }
            endpoint = self.config.base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
            counted_request = Request(
                endpoint,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer " + self.config.api_key,
                    "Content-Type": "application/json",
                },
            )
            with urlopen(counted_request, timeout=min(30, self.config.timeout_s)) as counted:
                prompt_count = int(json.load(counted)["count"])
            # Recovery uses the ordinary combined reasoning/action allowance.
            combined_cap = (
                min(requested_max_tokens, 1000)
                if _DIRECTOR_RECOVERY.get()
                else requested_max_tokens
            )
            request["max_tokens"] = director_static_budget(prompt_count, combined_cap)
        with _capture_request_events(request_events, role=role):
            completion = _openai_completion_create(
                self.client, self.config, request, deadline
            )
        generation_attempts = []

        def record_generation(completed, budget):
            usage = completed.usage
            generation_attempts.append(
                {
                    "max_output_tokens": budget,
                    "finish_reason": completed.choices[0].finish_reason,
                    "token_in": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "token_out": int(getattr(usage, "completion_tokens", 0) or 0),
                }
            )

        record_generation(completion, request.get("max_tokens"))
        selected_request = request
        choice = completion.choices[0]
        raw_reasoning_text, raw_action_text = response_policy_parts(
            choice.message,
            thinking_prefilled=self.config.request_profile == "qwen" and thinking_enabled,
        )
        qwen_thinking_boundary_restored = False
        if (
            role == "graph-director"
            and self.config.request_profile == "qwen"
            and thinking_enabled
            and not raw_action_text.strip()
            and not _openai_action_calls(choice.message)
        ):
            recovered_parts = recover_qwen_policy_parts(
                choice.message,
                completion_token_ids=getattr(choice, "token_ids", None) or (),
                content_logprobs=(
                    getattr(getattr(choice, "logprobs", None), "content", None) or ()
                ),
            )
            if recovered_parts is not None:
                raw_reasoning_text, raw_action_text = recovered_parts
                qwen_thinking_boundary_restored = True
        split_director_action_retry = False
        if (
            role == "graph-director"
            and self.config.request_profile == "qwen"
            and thinking_enabled
            and raw_reasoning_text.strip()
            and not raw_action_text.strip()
            and not _openai_action_calls(choice.message)
            # A thinking response without a visible action cannot be sent to
            # Canvas: the strict policy parser quite correctly turns it into
            # ``{"action":"invalid"}``.
        ):
            # Qwen3.5 can stop after writing the requested JSON inside its
            # private reasoning channel.  That is not a typed Director action:
            # replay the same authoritative Canvas prompt with thinking off so
            # the model emits only the externally visible JSON action.  Keep
            # the first call's reasoning for inference audit, but mark the
            # resulting split trace non-trainable below because its token IDs
            # and log-probabilities do not form one contiguous policy sample.
            action_retry_request = dict(request)
            action_retry_extra = dict(action_retry_request.get("extra_body", {}))
            action_retry_template = dict(action_retry_extra.get("chat_template_kwargs", {}))
            action_retry_template["enable_thinking"] = False
            action_retry_template.pop("thinking_budget", None)
            action_retry_extra["chat_template_kwargs"] = action_retry_template
            action_retry_request["extra_body"] = action_retry_extra
            action_retry_request["max_tokens"] = min(
                1024, int(action_retry_request.get("max_tokens", requested_max_tokens))
            )
            with _capture_request_events(request_events, role=role):
                action_retry = _openai_completion_create(
                    self.client, self.config, action_retry_request, deadline
                )
            record_generation(action_retry, action_retry_request["max_tokens"])
            retry_choice = action_retry.choices[0]
            retry_reasoning, retry_action = response_policy_parts(
                retry_choice.message,
                thinking_prefilled=False,
            )
            if retry_action.strip() or _openai_action_calls(retry_choice.message):
                split_director_action_retry = True
                qwen_thinking_boundary_restored = False
                completion, choice = action_retry, retry_choice
                selected_request = action_retry_request
                raw_reasoning_text = raw_reasoning_text + retry_reasoning
                raw_action_text = retry_action
        text = (
            raw_action_text.strip()
            if qwen_thinking_boundary_restored
            else response_content(
                choice.message,
                enable_thinking=thinking_enabled and not split_director_action_retry,
                reasoning_fallback=self.config.request_profile != "qwen",
            )
        )
        native_calls = _openai_action_calls(choice.message)
        if (
            choice.finish_reason == "length"
            and not native_calls
            and role not in {"graph-director", "proposer"}
            and not _FINALIZATION_REQUEST.get()
        ):
            retry_assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": text,
            }
            # Replay the provider-authored reasoning field with the truncated
            # assistant turn, in whichever spelling the endpoint returned.
            provider_message = _model_dump(choice.message)
            for reasoning_field in ("reasoning_content", "reasoning_text"):
                reasoning_value = provider_message.get(
                    reasoning_field, getattr(choice.message, reasoning_field, None)
                )
                if reasoning_value is not None:
                    retry_assistant_message[reasoning_field] = reasoning_value
            retry_messages = [
                *request_messages,
                retry_assistant_message,
                {
                    "role": "user",
                    "content": (
                        "The response was truncated, not an environment termination. "
                        "Continue with a concise legal tool call of your choice, or return "
                        "the required final JSON if you choose to finish. Do not repeat analysis."
                        if actions
                        else "The response was truncated. Return only the required short JSON object."
                    ),
                },
            ]
            retry_request: dict[str, Any] = {
                "model": role_config.model,
                "messages": retry_messages,
                "temperature": sent_temperature,
                "top_p": role_config.top_p,
            }
            if response_json_schema is not None:
                retry_request["response_format"] = request["response_format"]
            retry_budget = requested_max_tokens if actions else min(512, requested_max_tokens)
            retry_request["max_tokens"] = retry_budget
            if role_config.reasoning_effort is not None:
                retry_request["reasoning_effort"] = role_config.reasoning_effort
            if self.config.request_profile == "qwen":
                retry_request["extra_body"] = {
                    **qwen_request_extra(enable_thinking=False),
                    **({"top_k": role_config.top_k} if role_config.top_k is not None else {}),
                }
            if actions and not qwen_text_actions:
                tools = [_openai_tool(action) for action in actions]
                retry_request.update(tools=tools, tool_choice="auto")
                retry_request["parallel_tool_calls"] = False
            with _capture_request_events(request_events, role=role):
                retry = _openai_completion_create(
                    self.client, self.config, retry_request, deadline
                )
            record_generation(retry, retry_request.get("max_tokens"))
            retry_choice = retry.choices[0]
            retry_text = response_content(
                retry_choice.message,
                enable_thinking=False,
                reasoning_fallback=self.config.request_profile != "qwen",
            )
            if retry_text or _openai_action_calls(retry_choice.message):
                completion, choice, text = retry, retry_choice, retry_text
                selected_request = retry_request
                raw_reasoning_text, raw_action_text = response_policy_parts(
                    choice.message,
                    thinking_prefilled=self.config.request_profile == "qwen" and thinking_enabled,
                )
                native_calls = _openai_action_calls(choice.message)
        content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None) or []
        sampled_log_probs = tuple(
            float(item.logprob)
            for item in content_logprobs
            if getattr(item, "logprob", None) is not None
        )
        sampled_token_ids = tuple(
            int(value) for value in (getattr(choice, "token_ids", None) or ())
        )
        prompt_token_ids = tuple(
            int(value) for value in (getattr(completion, "prompt_token_ids", None) or ())
        )
        exact_sample_trace = bool(
            not split_director_action_retry
            and
            prompt_token_ids
            and sampled_token_ids
            and sampled_log_probs
            and len(sampled_token_ids) == len(sampled_log_probs)
        )
        provider_model = str(getattr(completion, "model", "") or "").strip()
        _attach_provider_usage(
            request_events,
            getattr(completion, "usage", None),
            surface="chat_completions",
            model=provider_model,
            effort=role_config.reasoning_effort,
        )
        result = LLMResponse(
            text=text,
            model=provider_model or role_config.model,
            action_calls=native_calls,
            assistant_message=_canonical_assistant_message(
                text=text,
                calls=native_calls,
                provider="openai",
                provider_payload=_model_dump(choice.message),
            ),
            token_in=sum(item["token_in"] for item in generation_attempts),
            token_out=sum(item["token_out"] for item in generation_attempts),
            metadata={
                "role": role,
                "api_surface": "chat_completions",
                "qwen_text_actions": qwen_text_actions,
                "finish_reason": choice.finish_reason,
                "enable_thinking": thinking_enabled,
                "reasoning_effort": role_config.reasoning_effort,
                "temperature_sent": (sent_temperature),
                "max_output_tokens_sent": selected_request.get("max_tokens"),
                "requested_max_output_tokens": requested_max_tokens,
                "requested_model": role_config.model,
                "provider_model": provider_model,
                "route": self.config.route_name,
                "model_attestation_source": (
                    "provider_response" if provider_model else "request_fallback"
                ),
                "backend_request_events": request_events,
                "generation_attempts": generation_attempts,
                "split_director_action_retry": split_director_action_retry,
                "qwen_thinking_boundary_restored": qwen_thinking_boundary_restored,
                "qwen_thinking_boundary_source": (
                    "provider_token_logprob_bytes"
                    if qwen_thinking_boundary_restored
                    else "provider_message"
                ),
                "runtime_managed_finalization": _FINALIZATION_REQUEST.get(),
                "sampling_seed": self.config.sampling_seed,
                "request_seed": selected_request.get("seed"),
            },
            raw_reasoning_text=raw_reasoning_text,
            raw_action_text=raw_action_text,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=sampled_token_ids,
            behavior_log_probs=sampled_log_probs,
            token_provenance=(
                "provider_prompt_and_completion_token_ids_and_logprobs"
                if exact_sample_trace
                else "text_only"
            ),
            training_eligible=bool((raw_reasoning_text or raw_action_text) and exact_sample_trace),
        )
        if deadline is not None:
            deadline.check(f"{role}_request_complete")
            if role == "worker":
                deadline.mark_progress("worker_response")
        return result

    def choose_binary(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        token_ids: dict[str, int],
    ) -> BinaryChoiceResponse:
        """Sample one token-ID-constrained choice and retain its true two-way policy."""

        if set(token_ids) != {"off", "on"} or token_ids["off"] == token_ids["on"]:
            raise BinaryChoiceUnavailable("off/on tokenizer IDs are missing or invalid")
        role_config = self.config.roles.get(role, self.config.roles.get("worker"))
        if role_config is None:
            raise BinaryChoiceUnavailable(f"model role is not configured: {role}")
        if self.config.request_profile != "qwen":
            raise BinaryChoiceUnavailable(
                "this route has no verified off/on guided-choice and logprobs contract"
            )
        deadline = getattr(self, "rollout_deadline", None)
        if deadline is not None:
            deadline.check(f"{role}_binary_request_start")
            scope_request = getattr(deadline, "for_request", None)
            if callable(scope_request):
                deadline = scope_request(role, self.config.timeout_s)
        request_messages = _openai_messages(messages)
        if role_config.system_prompt:
            request_messages = [
                {"role": "system", "content": role_config.system_prompt},
                *request_messages,
            ]
        request: dict[str, Any] = {
            "model": role_config.model,
            "messages": request_messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 0,
            "extra_body": {
                **qwen_request_extra(
                    enable_thinking=False,
                    director_timeline=(
                        role == "graph-director"
                        and director_context_mode() in TIMELINE_CONTEXT_MODES
                    ),
                ),
                # vLLM's string-choice grammar can start with a sub-token such
                # as ``o`` and then stop at max_tokens=1. Restricting the
                # support by the tokenizer-attested IDs makes the sampled
                # action exactly one of the two trainable tokens. Explicit ID
                # logprobs avoid relying on an arbitrary top-k cutoff.
                "allowed_token_ids": [int(token_ids["off"]), int(token_ids["on"])],
                "logprob_token_ids": [int(token_ids["off"]), int(token_ids["on"])],
                "return_token_ids": True,
            },
        }
        if request["extra_body"]["chat_template_kwargs"].get("director_append_only"):
            # Append-only histories can reach the context limit during a
            # relation choice too. Preserve history and terminate cleanly.
            body = {
                "model": role_config.model,
                "messages": request_messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": request["extra_body"]["chat_template_kwargs"],
            }
            endpoint = self.config.base_url.rstrip("/").removesuffix("/v1") + "/tokenize"
            counted_request = Request(
                endpoint,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": "Bearer " + self.config.api_key,
                    "Content-Type": "application/json",
                },
            )
            with urlopen(counted_request, timeout=min(30, self.config.timeout_s)) as counted:
                director_static_budget(int(json.load(counted)["count"]), 1)
        if self.config.sampling_seed is not None:
            request["seed"] = int(self.config.sampling_seed)
        request_events: list[dict[str, Any]] = []
        with _capture_request_events(request_events, role=role):
            completion = _openai_completion_create(
                self.client, self.config, request, deadline
            )
        choice = completion.choices[0]
        sampled = str(getattr(choice.message, "content", "") or "")
        if sampled not in {"off", "on"}:
            raise BinaryChoiceUnavailable(
                f"token-constrained binary response was not exactly off/on: {sampled!r}"
            )
        sampled_token_ids = tuple(
            int(value) for value in (getattr(choice, "token_ids", None) or ())
        )
        if sampled_token_ids != (int(token_ids[sampled]),):
            raise BinaryChoiceUnavailable(
                "provider binary token ID does not match the sampled off/on choice"
            )
        prompt_token_ids = tuple(
            int(value) for value in (getattr(completion, "prompt_token_ids", None) or ())
        )
        if not prompt_token_ids:
            raise BinaryChoiceUnavailable("provider binary response lacks prompt token IDs")
        content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None) or []
        if len(content_logprobs) != 1:
            raise BinaryChoiceUnavailable(
                "binary response did not expose exactly one token logprob"
            )
        top_items = getattr(content_logprobs[0], "top_logprobs", None) or []
        observed: dict[str, float] = {}
        for item in top_items:
            token = str(getattr(item, "token", ""))
            if token in {"off", "on"}:
                observed[token] = float(item.logprob)
        if set(observed) != {"off", "on"}:
            raise BinaryChoiceUnavailable(
                "provider did not return both constrained off/on log probabilities"
            )
        raw_probabilities = {key: math.exp(value) for key, value in observed.items()}
        constrained_mass = sum(raw_probabilities.values())
        if not math.isfinite(constrained_mass) or constrained_mass <= 0.0:
            raise BinaryChoiceUnavailable(
                "off/on log probabilities have invalid unconstrained support mass"
            )
        probabilities = {key: value / constrained_mass for key, value in raw_probabilities.items()}
        normalized_log_probabilities = {
            key: math.log(value) for key, value in probabilities.items()
        }
        usage = completion.usage
        provider_model = str(getattr(completion, "model", "") or "").strip()
        _attach_provider_usage(
            request_events,
            usage,
            surface="chat_completions",
            model=provider_model,
            effort=role_config.reasoning_effort,
        )
        result = BinaryChoiceResponse(
            choice=sampled,
            model=provider_model or role_config.model,
            probabilities=probabilities,
            log_probabilities=normalized_log_probabilities,
            token_ids={key: int(value) for key, value in token_ids.items()},
            prompt_token_ids=prompt_token_ids,
            token_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            token_out=int(getattr(usage, "completion_tokens", 0) or 0),
            metadata={
                "role": role,
                "requested_model": role_config.model,
                "provider_model": provider_model,
                "api_surface": "chat_completions",
                "constraint": "allowed_token_ids",
                "logprob_surface": "logprob_token_ids",
                "temperature_sent": 1.0,
                "top_p_sent": 1.0,
                "top_k_sent": None,
                "enable_thinking": False,
                "sampling_seed": self.config.sampling_seed,
                "request_seed": request.get("seed"),
                # vLLM exposes requested token-ID logprobs under the original
                # vocabulary softmax, even though sampling applies the allowed
                # token mask. Renormalizing these two entries is the exact
                # post-mask binary policy used for PPO.
                "unconstrained_support_mass": constrained_mass,
                "constrained_probability_mass": 1.0,
                "backend_request_events": request_events,
            },
        )
        if deadline is not None:
            deadline.check(f"{role}_binary_request_complete")
        return result

    def choose_label(
        self,
        messages: list[dict[str, Any]],
        *,
        role: str,
        token_ids: dict[str, int],
        seed: int,
        temperature: float = 1.0,
    ) -> LabelChoiceResponse:
        """Sample one label token under the same ID constraint as ``choose_binary``.

        The request seed is per call, so two choices on one frozen list are
        independent samples rather than replays of ``config.sampling_seed``.
        """

        support = [int(value) for value in token_ids.values()]
        if len(support) < 2 or len(set(support)) != len(support):
            raise LabelChoiceUnavailable("label tokenizer IDs are missing or not distinct")
        role_config = self.config.roles.get(role, self.config.roles.get("worker"))
        if role_config is None:
            raise LabelChoiceUnavailable(f"model role is not configured: {role}")
        if self.config.request_profile != "qwen":
            raise LabelChoiceUnavailable(
                "label policy audit requires the qwen chat-completions logprobs surface"
            )
        request_messages = _openai_messages(messages)
        if role_config.system_prompt:
            request_messages = [
                {"role": "system", "content": role_config.system_prompt},
                *request_messages,
            ]
        request: dict[str, Any] = {
            "model": role_config.model,
            "messages": request_messages,
            "temperature": float(temperature),
            "top_p": 1.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 0,
            "seed": int(seed),
            "extra_body": {
                **qwen_request_extra(enable_thinking=False),
                "top_k": -1,
                "allowed_token_ids": support,
                "logprob_token_ids": support,
                "return_token_ids": True,
            },
        }
        request_events: list[dict[str, Any]] = []
        with _capture_request_events(request_events, role=role):
            completion = _openai_completion_create(
                self.client, self.config, request, None
            )
        choice = completion.choices[0]
        sampled = str(getattr(choice.message, "content", "") or "")
        if sampled not in token_ids:
            raise LabelChoiceUnavailable(
                f"token-constrained label response was not a listed label: {sampled!r}"
            )
        sampled_token_ids = tuple(
            int(value) for value in (getattr(choice, "token_ids", None) or ())
        )
        if sampled_token_ids != (int(token_ids[sampled]),):
            raise LabelChoiceUnavailable("provider label token ID does not match the label")
        prompt_token_ids = tuple(
            int(value) for value in (getattr(completion, "prompt_token_ids", None) or ())
        )
        if not prompt_token_ids:
            raise LabelChoiceUnavailable("provider label response lacks prompt token IDs")
        content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None) or []
        if len(content_logprobs) != 1:
            raise LabelChoiceUnavailable("label response did not expose exactly one token logprob")
        observed = {
            str(getattr(item, "token", "")): float(item.logprob)
            for item in getattr(content_logprobs[0], "top_logprobs", None) or []
            if str(getattr(item, "token", "")) in token_ids
        }
        if set(observed) != set(token_ids):
            raise LabelChoiceUnavailable("provider did not return every constrained label logprob")
        # As in choose_binary: vLLM reports requested-ID logprobs under the full
        # vocabulary softmax; renormalizing over the allowed IDs gives the exact
        # post-mask policy that sampled the label.
        mass = sum(math.exp(value) for value in observed.values())
        if not math.isfinite(mass) or mass <= 0.0:
            raise LabelChoiceUnavailable("label log probabilities have invalid support mass")
        log_mass = math.log(mass)
        usage = completion.usage
        provider_model = str(getattr(completion, "model", "") or "").strip()
        _attach_provider_usage(
            request_events,
            usage,
            surface="chat_completions",
            model=provider_model,
            effort=role_config.reasoning_effort,
        )
        return LabelChoiceResponse(
            label=sampled,
            log_probabilities={key: value - log_mass for key, value in observed.items()},
            token_ids={key: int(value) for key, value in token_ids.items()},
            prompt_token_ids=prompt_token_ids,
            model=provider_model or role_config.model,
            token_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            token_out=int(getattr(usage, "completion_tokens", 0) or 0),
            metadata={
                "role": role,
                "requested_model": role_config.model,
                "provider_model": provider_model,
                "constraint": "allowed_token_ids",
                "logprob_surface": "logprob_token_ids",
                "temperature_sent": float(temperature),
                "top_p_sent": 1.0,
                "top_k_sent": -1,
                "enable_thinking": False,
                "sampling_seed": int(seed),
                "unconstrained_support_mass": mass,
                "backend_request_events": request_events,
            },
        )

def _text_action_messages(messages: Sequence[dict[str, Any]], actions: Sequence[ActionSpec]) -> list[dict[str, Any]]:
    """Use plain conversation items for gateways that prohibit native tools."""
    converted = []
    if actions:
        converted.append({"role": "developer", "content": (
            'To execute a local action, output only JSON: '
            '{"action_calls":[{"name":"ACTION_NAME","arguments":{}}]}. '
            'Choose one available action and conform to its parameters. '
            'Emit exactly one JSON object per response. After an action request, stop; '
            'never append another action object or a final answer in the same response. '
            'Do not invent execution results; wait for the subsequent observation. '
            'When finished, return the requested final answer. Available actions: '
            + json.dumps([_openai_tool(action)["function"] for action in actions])
        )})
    for message in _openai_messages(messages):
        role = message.get("role", "user")
        content = message.get("content") or ""
        calls = message.get("tool_calls", [])
        if calls:
            content = json.dumps({"action_calls": [
                {"name": call["function"]["name"],
                 "arguments": json.loads(call["function"]["arguments"])} for call in calls
            ]})
        if role == "tool":
            role = "user"
            content = "Local action observation: " + str(content)
        converted.append({"role": role, "content": content})
    return converted


def _openai_tool(action: ActionSpec) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": action.name,
        "description": action.description,
        "parameters": action.parameters,
    }
    if _strict_schema_compatible(action.parameters):
        function["strict"] = True
    return {
        "type": "function",
        "function": function,
    }


def _strict_schema_compatible(schema: object) -> bool:
    """Use provider strict mode only when every object schema satisfies its contract."""

    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if (
            not isinstance(properties, dict)
            or not isinstance(required, list)
            or set(properties) != set(required)
            or schema.get("additionalProperties") is not False
        ):
            return False
        return all(_strict_schema_compatible(value) for value in properties.values())
    if schema_type == "array":
        return _strict_schema_compatible(schema.get("items", {}))
    if isinstance(schema_type, (str, list)):
        return True
    return any(
        isinstance(schema.get(keyword), list)
        and all(_strict_schema_compatible(value) for value in schema[keyword])
        for keyword in ("anyOf", "oneOf")
    )


def _openai_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        provider_payload = message.get("_provider_payloads", {}).get("openai")
        if isinstance(provider_payload, dict):
            converted.append(dict(provider_payload))
            continue
        role = str(message.get("role", "user"))
        if role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("call_id", "")),
                    "content": str(message.get("content", "")),
                }
            )
            continue
        converted_message: dict[str, Any] = {
            "role": role,
            "content": message.get("content", ""),
        }
        calls = message.get("action_calls", [])
        if role == "assistant" and isinstance(calls, list) and calls:
            converted_message["tool_calls"] = [
                {
                    "id": str(call.get("call_id", "")),
                    "type": "function",
                    "function": {
                        "name": str(call.get("name", "")),
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
                for call in calls
                if isinstance(call, dict)
            ]
        converted.append(converted_message)
    return converted


def _openai_action_calls(message: Any) -> list[ActionCall]:
    normalized: list[ActionCall] = []
    for index, call in enumerate(getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        if function is None:
            continue
        raw_arguments = getattr(function, "arguments", {})
        try:
            arguments: Any = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except (TypeError, ValueError):
            arguments = raw_arguments
        normalized.append(
            ActionCall(
                call_id=str(getattr(call, "id", "") or f"openai-call-{index}"),
                name=str(getattr(function, "name", "")),
                arguments=arguments,
            )
        )
    return normalized


def _model_dump(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(exclude_none=True)
        if isinstance(payload, dict):
            return payload
    payload: dict[str, Any] = {
        "role": str(getattr(value, "role", "assistant")),
        "content": getattr(value, "content", None),
    }
    calls = getattr(value, "tool_calls", None)
    if calls:
        payload["tool_calls"] = []
        for call in calls:
            function = getattr(call, "function", None)
            payload["tool_calls"].append(
                {
                    "id": str(getattr(call, "id", "")),
                    "type": "function",
                    "function": {
                        "name": str(getattr(function, "name", "")),
                        "arguments": str(getattr(function, "arguments", "{}")),
                    },
                }
            )
    return payload


def _canonical_assistant_message(
    *,
    text: str,
    calls: Sequence[ActionCall],
    provider: str,
    provider_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "action_calls": [call.to_dict() for call in calls],
        "_provider_payloads": {provider: provider_payload},
    }
