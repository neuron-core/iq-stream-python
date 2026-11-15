"""How a run ended, as ``publish()`` reports it, and the close line that follows from it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from iq_stream import Outcome, PublishResult
from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError
from workflows.events import StopEvent, WorkflowCancelledEvent, WorkflowFailedEvent, WorkflowTimedOutEvent

RunStatus = Literal["completed", "failed", "timed_out", "cancelled"]


@dataclass(frozen=True)
class RunOutcome:
    """How a run ended and what reached the publisher."""

    #: ``completed``, ``failed`` (a step raised for good), ``timed_out`` or ``cancelled``.
    status: RunStatus
    #: The handler's run id.
    run_id: str
    #: What ``await handler`` returned: for an agent, its ``AgentOutput``; ``None`` when the run did not complete.
    result: Any
    #: The terminal event of the stream, or the handler's stop event when the stream had none.
    stop_event: StopEvent | None
    #: What ``await handler`` raised: the step's exception, ``WorkflowTimeoutError``, ``WorkflowCancelledByUser``.
    error: BaseException | None
    #: Frames the publisher accepted.
    frames: int
    #: False once the publisher gave up: the browser was told ``iq:fail`` and loads the answer from the application.
    published: bool
    #: The server's answer to the close line; ``None`` before the close (in ``before_close``) and when no close was
    #: sent because the publisher had given up.
    publish_result: PublishResult | None = None


def classify(terminal: StopEvent | None, error: BaseException | None) -> RunStatus:
    """The status from the terminal stream event and what awaiting the handler raised."""
    if isinstance(terminal, WorkflowTimedOutEvent) or isinstance(error, WorkflowTimeoutError):
        return "timed_out"
    if isinstance(terminal, WorkflowCancelledEvent) or isinstance(
        error, (WorkflowCancelledByUser, asyncio.CancelledError)
    ):
        return "cancelled"
    if isinstance(terminal, WorkflowFailedEvent) or error is not None:
        return "failed"
    return "completed"


def close_line(status: RunStatus, close_on_cancel: Outcome) -> Outcome:
    """``end`` for a completed run, ``fail`` for a failed or timed-out one, ``close_on_cancel`` for a cancelled one."""
    if status == "completed":
        return "end"
    if status == "cancelled":
        return close_on_cancel
    return "fail"
