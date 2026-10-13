from __future__ import annotations

from .answer_submission import submission_contract
from .observability import TaskSpec


def solver_task_text(task: TaskSpec) -> str:
    """Render the original question with the submission contract."""

    # The QA answer interface is independent of optional model-based formatting.
    # Apply this on primary and graph re-execution paths alike, without giving
    # intermediate Workers an obligation to solve the complete task.
    # Final submission formatting is injected by the runtime only for the
    # selected output Agent.  The Canvas/Director may still request it
    # explicitly, but intermediate QA Workers should receive only the task.
    contract = submission_contract(task)
    task_text = task.prompt
    return f"{task_text}\n\n{contract}" if contract else task_text
