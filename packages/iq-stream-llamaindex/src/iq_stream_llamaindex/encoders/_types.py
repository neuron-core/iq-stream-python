"""The optional packages' event classes, looked up only once something has loaded them: an event can only be an
instance of a class whose module is loaded, so a workflow that never touches llama-index-core or ag-ui-protocol does
not pay for importing them."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, cast

from workflows.events import WorkflowFailedEvent

if TYPE_CHECKING:
    from llama_index.core.agent.workflow.workflow_events import ToolCall, ToolCallResult


def agent_events():
    """``llama_index.core.agent.workflow.workflow_events``, if loaded."""
    if "llama_index.core.agent.workflow.workflow_events" not in sys.modules:
        return None
    from llama_index.core.agent.workflow import workflow_events

    return workflow_events


def agui_events():
    """``ag_ui.core.events``, if ag-ui-protocol is loaded."""
    if "ag_ui" not in sys.modules:
        return None
    from ag_ui.core import events

    return events


def failure(event: WorkflowFailedEvent) -> tuple[str, str]:
    """The failed step's exception, as its type name and its message. Newer workflows releases carry the exception
    itself; 2.14 to 2.2x carried ``exception_type`` and ``exception_message``."""
    exception: object = getattr(event, "exception", None)
    if isinstance(exception, BaseException):
        return type(exception).__name__, str(exception)
    kind = str(getattr(event, "exception_type", "") or "")
    return kind.rpartition(".")[2], str(getattr(event, "exception_message", "") or "")


def tool_kwargs(event: ToolCall | ToolCallResult) -> dict[str, Any]:
    """A tool call's arguments, which llama-index-core types as a bare ``dict``."""
    return cast("dict[str, Any]", event.tool_kwargs)  # pyright: ignore[reportUnknownMemberType]
