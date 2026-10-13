"""Endpoint sharding and bounded, request-local failover."""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .backend_failures import (
    BackendFailureClassification,
    BackendRequestError,
    classify_backend_failure,
)
from .llm import _logical_request_budget_s, endpoint_failover_scope


def _merge_request_events(*groups):
    """Preserve physical attempts across failover without counting an event twice."""
    result = []
    seen = set()
    for group in groups:
        for event in group:
            if not isinstance(event, dict):
                continue
            identity = event.get("event_id")
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            result.append(dict(event))
    return result


class EndpointPoolBackend:
    def __init__(
        self,
        name: str,
        members: dict[str, Any],
        state_dir: Path,
        *,
        pool_retry_attempts: int = 0,
        retry_backoff_s: float = 0.0,
        member_queue_wait_s: float = 0.5,
    ):
        if not members:
            raise ValueError("endpoint pool must not be empty")
        if pool_retry_attempts < 0 or retry_backoff_s < 0 or member_queue_wait_s < 0:
            raise ValueError("endpoint pool retry settings must be non-negative")
        self.name = name
        self.members = tuple(members.items())
        self.pool_retry_attempts = int(pool_retry_attempts)
        self.retry_backoff_s = float(retry_backoff_s)
        self.member_queue_wait_s = float(member_queue_wait_s)
        self.config = self.members[0][1].config
        signature = repr([(key, backend.config.base_url) for key, backend in self.members])
        self.counter_path = state_dir / (
            hashlib.sha256(signature.encode()).hexdigest() + ".counter"
        )
        self._deadline = ContextVar(f"endpoint_pool_deadline_{id(self)}", default=None)

    def _audit(self, event):
        # Persist failures even when no completed Worker result reaches the runner.
        with self.counter_path.with_suffix(".events.jsonl").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def generate(self, messages, **kwargs):
        self.counter_path.parent.mkdir(parents=True, exist_ok=True)
        with self.counter_path.open("a+") as counter:
            fcntl.flock(counter, fcntl.LOCK_EX)
            counter.seek(0)
            value = int(counter.read() or "0")
            counter.seek(0)
            counter.truncate()
            counter.write(str(value + 1))
            counter.flush()
        ordered = (
            self.members[value % len(self.members) :] + self.members[: value % len(self.members)]
        )
        deadline = self._deadline.get()
        started = time.monotonic()
        first_config = ordered[0][1].config
        request_deadline = deadline
        if deadline is not None:
            deadline.check("endpoint_pool_start")
            scope_request = getattr(deadline, "for_request", None)
            if callable(scope_request):
                request_deadline = scope_request(
                    kwargs.get("role", "worker"), first_config.timeout_s
                )
        # Bound a pooled call by its slowest configured member. Using only the
        # first member's timeout leaves no time to fail over when that member
        # consumes its full timeout.
        member_budgets = [
            _logical_request_budget_s(backend.config, request_deadline)
            for _, backend in ordered
        ]
        budget = max(member_budgets)
        end = started + budget
        request_id = uuid.uuid4().hex
        attempts = []
        failures = []
        failed_request_events = []
        last_error = None
        for pool_round in range(self.pool_retry_attempts + 1):
            round_members = (
                ordered[pool_round % len(ordered) :] + ordered[: pool_round % len(ordered)]
            )
            for member_index, (key, backend) in enumerate(round_members):
                # Failed waits do not consume the next member's request budget.
                if getattr(deadline, "exclude_failed_request_time", False):
                    end = time.monotonic() + _logical_request_budget_s(
                        backend.config, request_deadline
                    )
                if deadline is not None:
                    deadline.check("endpoint_pool_failover")
                if time.monotonic() >= end and last_error is not None:
                    break
                attempt_started = time.monotonic()
                request_remaining_s = max(0.0, end - attempt_started)
                # Without failed-request time credit, share
                # the bounded pool window across the remaining endpoints in
                # this round. A quick queue failure leaves almost all of the
                # window for the next member; a stalled request cannot consume
                # the fallback member's reserved time.
                if getattr(deadline, "exclude_failed_request_time", False):
                    attempt_end = end
                else:
                    members_left = len(round_members) - member_index
                    attempt_end = attempt_started + request_remaining_s / members_left
                # Only completed canonical messages cross endpoints. Tools execute
                # outside generate(), so failover cannot replay an environment action.
                replay = [
                    {
                        k: v
                        for k, v in message.items()
                        if k not in {"_provider_payloads", "_endpoint_pool_member"}
                    }
                    if message.get("_endpoint_pool_member") != key
                    else {k: v for k, v in message.items() if k != "_endpoint_pool_member"}
                    for message in messages
                ]
                event = dict(
                    request_id=request_id,
                    logical_route=self.name,
                    endpoint_pool_member=key,
                    pool_round=pool_round + 1,
                    attempt=len(attempts) + 1,
                    request_role=kwargs.get("role", ""),
                    timestamp=time.time(),
                    request_remaining_s=request_remaining_s,
                    endpoint_attempt_budget_s=max(0.0, attempt_end - attempt_started),
                )
                try:
                    # A saturated member must not hold the logical request in
                    # its local queue while another pool member has capacity.
                    # Give short bursts a bounded grace period, then surface a
                    # retryable local queue timeout and immediately fail over.
                    with endpoint_failover_scope(
                        attempt_end, queue_wait_cap_s=self.member_queue_wait_s
                    ):
                        response = backend.generate(replay, **kwargs)
                except Exception as exc:
                    current_events = _merge_request_events(getattr(exc, "request_events", ()))
                    failed_request_events = _merge_request_events(failed_request_events, current_events)
                    failure = classify_backend_failure(exc, stage="endpoint_pool", route=key)
                    if failure.backend_failure and hasattr(deadline, "record_failed_request"):
                        deadline.record_failed_request(attempt_started, time.monotonic())
                    event.update(
                        status="failed",
                        elapsed_s=time.monotonic() - attempt_started,
                        failure=failure.to_dict(),
                    )
                    if current_events:
                        event["backend_request_events"] = current_events
                    attempts.append(event)
                    failures.append(failure)
                    self._audit(event)
                    exc.endpoint_pool_attempts = list(attempts)
                    last_error = exc
                    # Invalid requests, environment failures and controller deadlines
                    # are shared call failures, not endpoint availability failures.
                    if not failure.backend_failure or not (
                        failure.retryable or failure.disable_route
                    ):
                        if failed_request_events:
                            exc.request_events = list(failed_request_events)
                        raise
                    continue
                event.update(status="success", elapsed_s=time.monotonic() - attempt_started)
                attempts.append(event)
                self._audit(event)
                if getattr(response, "assistant_message", None) is not None:
                    response.assistant_message["_endpoint_pool_member"] = key
                response.metadata.update(
                    logical_route=self.name,
                    endpoint_pool_member=key,
                    endpoint_pool_attempts=attempts,
                    endpoint_pool_failovers=len(attempts) - 1,
                    endpoint_pool_retries=pool_round,
                )
                if failed_request_events:
                    response.metadata["backend_request_events"] = _merge_request_events(
                        failed_request_events, response.metadata.get("backend_request_events", ())
                    )
                return response
            if pool_round >= self.pool_retry_attempts or last_error is None:
                break
            delay_s = self.retry_backoff_s * 2**pool_round
            if time.monotonic() + delay_s >= end:
                break
            if delay_s:
                time.sleep(delay_s)
        assert last_error is not None
        kinds = {failure.kind for failure in failures}
        attempted_members = sorted({event["endpoint_pool_member"] for event in attempts})
        all_members_attempted = len(attempted_members) == len(self.members)
        budget_exhausted = time.monotonic() >= end
        classification = BackendFailureClassification(
            backend_failure=True,
            origin="endpoint_pool",
            kind=next(iter(kinds)) if len(kinds) == 1 else "endpoint_pool_exhausted",
            retryable=False,
            counts_toward_route_circuit=True,
            # A request deadline can prevent trying the other healthy members.
            # Do not disable the entire logical route on that evidence alone.
            disable_route=all_members_attempted,
            stage="endpoint_pool",
            route=self.name,
            exception_type="EndpointPoolExhausted",
            message=(
                f"{len(attempted_members)} of {len(self.members)} endpoints attempted; "
                f"{len(attempts)} failed attempts across "
                f"{max(event['pool_round'] for event in attempts)} pool rounds; "
                f"request budget exhausted={budget_exhausted}"
            ),
        )
        terminal_event = {
            "schema_version": 1,
            "event": "backend_request_failure",
            **classification.to_dict(),
            "attempt": len(attempts),
            "will_retry": False,
            "endpoint_pool_attempts": len(attempts),
            "endpoint_pool_members_attempted": attempted_members,
            "endpoint_pool_members_total": len(self.members),
            "request_budget_exhausted": budget_exhausted,
        }
        error = BackendRequestError(
            classification, request_events=[*failed_request_events, terminal_event]
        )
        error.endpoint_pool_attempts = list(attempts)
        raise error from last_error

    def set_deadline_context(self, deadline):
        self._deadline.set(deadline)
        for _, backend in self.members:
            backend.set_deadline_context(deadline)
