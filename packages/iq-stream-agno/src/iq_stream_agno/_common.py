"""What the tee and its two encoders share."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from iq_stream import IQError

Protocol = Literal["agno", "agui"]
"""The wire format of the frames: Agno's own event JSON, or AG-UI events."""

#: One frame to send: its SSE event name (``None`` for an unnamed frame) and its data.
Frame = tuple[str | None, str]

#: Whether a frame fits the publisher's line limit.
Fits = Callable[[str, str | None], bool]

#: Where an adapter-level problem goes: a dropped frame, an event that could not be encoded.
Report = Callable[[IQError], None]

#: What a browser reads instead of an error's own message, unless ``expose_errors`` is set.
MASKED_ERROR = "An error occurred."

#: The server's pattern for event names. A name outside it is sent unnamed rather than refused.
EVENT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")

#: Events that end a run as failed, cancelled or paused, when they carry the root run's id.
ROOT_ERRORS = frozenset({"RunError", "TeamRunError", "WorkflowError"})
ROOT_CANCELS = frozenset({"RunCancelled", "TeamRunCancelled", "WorkflowCancelled"})
ROOT_PAUSES = frozenset({"RunPaused", "TeamRunPaused", "WorkflowPaused"})

#: Error events and the member that carries the error's text; both encoders mask it.
ERROR_TEXT = {
    "RunError": "content",
    "TeamRunError": "content",
    "WorkflowError": "error",
    "StepError": "error",
    "ToolCallError": "error",
    "TeamToolCallError": "error",
}
#: Events whose ``tool`` holds a failed tool's error text as its result.
TOOL_EVENTS = frozenset({"ToolCallCompleted", "TeamToolCallCompleted", "ToolCallError", "TeamToolCallError"})


@dataclass(frozen=True)
class AguiRequest:
    """What an AG-UI request fixes for the events published in answer to it: the ids that ``RUN_STARTED`` and
    ``RUN_FINISHED`` carry (the client's, even when Agno resumes a paused run under that run's own id), and the
    request's state, which Agno's converter snapshots and diffs as the run changes it."""

    thread_id: str
    run_id: str
    state: dict[str, Any] | None = None


def nested(event: object) -> str | None:
    """The run an event's run is nested in: a team member's parent run, a workflow step's workflow run. ``None``
    for the outermost run's own events."""
    for key in ("parent_run_id", "workflow_run_id"):
        value: object = getattr(event, key, None)
        if isinstance(value, str) and value:
            return value
    return None


def event_name(event: object) -> str:
    """The Agno name of an event: its ``event`` field, which a custom event may set to an enum or leave empty."""
    value: object = getattr(event, "event", None)
    if isinstance(value, Enum):
        value = value.value
    return value if isinstance(value, str) else ""


def longest_prefix(text: str, fits: Callable[[str], bool]) -> int:
    """The length of the longest prefix of ``text`` that ``fits``, or -1 when not even the empty one does. The size
    of a frame grows with the prefix, so a binary search finds it in a few encodings."""
    if not fits(""):
        return -1
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(text[:middle]):
            low = middle
        else:
            high = middle - 1
    return low
