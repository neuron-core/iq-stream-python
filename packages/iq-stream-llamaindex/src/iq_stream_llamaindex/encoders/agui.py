"""``AGUIEncoder``: a run's events as AG-UI events, built with the ``ag_ui.core`` models (the ``agui`` extra)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from workflows.events import (
    Event,
    InputRequiredEvent,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowTimedOutEvent,
)

from ._types import agent_events, failure, tool_kwargs
from .base import (
    DEFAULT_MAX_FIELD_BYTES,
    MASKED_ERROR,
    Fits,
    Frame,
    bounded_json,
    cut,
    dumps,
    event_name,
    fields,
    shrink_strings,
    split_text,
)

if TYPE_CHECKING:
    from ag_ui.core import BaseEvent
    from llama_index.core.agent.workflow.workflow_events import AgentOutput, AgentStream, ToolCallResult

_ELLIPSIS = "…"
TIMED_OUT = "The run timed out."
# Events whose delta a client concatenates: one that is too long is sent as several.
_DELTAS = frozenset(
    {
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_CHUNK",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_CHUNK",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_CHUNK",
    }
)
# Events whose text may be cut to fit; anything else that does not fit (a snapshot) is dropped.
_CUTTABLE = frozenset({"TOOL_CALL_RESULT", "CUSTOM", "RUN_ERROR"})
_KEEP = frozenset({"type", "messageId", "toolCallId", "toolCallName", "name", "code", "role", "threadId", "runId"})


def _ag_ui():
    try:
        from ag_ui.core import events
    except ImportError as error:
        raise ImportError("AGUIEncoder needs ag-ui-protocol: pip install 'iq-stream-llamaindex[agui]'") from error
    return events


class AGUIEncoder:
    """Publishes a run as AG-UI events, one per frame, unnamed, as AG-UI's own SSE transport sends them; the browser
    reads them with ``StreamAgent`` from ``@neuron-core/iq-stream/ag-ui``.

    The agent events of llama-index-core are mapped (text and reasoning deltas, tool calls and results, one step per
    agent of an ``AgentWorkflow``); events that already are AG-UI events (from ``AGUIChatWorkflow``) pass through
    unchanged; any other event becomes a ``CUSTOM`` event named after its class, unless ``custom_events`` is false.
    The run opens with ``RUN_STARTED`` and ends with ``RUN_FINISHED`` (with the cancelled outcome for a cancelled run)
    or ``RUN_ERROR``. Tool arguments and results longer than ``max_field_bytes`` are cut, with an ellipsis. An
    exception's text is masked as ``"An error occurred."`` unless ``expose_errors`` is true.
    """

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str | None = None,
        emit_steps: bool = True,
        custom_events: bool = True,
        max_field_bytes: int = DEFAULT_MAX_FIELD_BYTES,
        expose_errors: bool = False,
    ) -> None:
        if max_field_bytes < 1:
            raise ValueError("max_field_bytes must be positive")
        self._ag = _ag_ui()
        self._thread_id = thread_id
        self._run_id = run_id
        self._emit_steps = emit_steps
        self._custom_events = custom_events
        self._max_field_bytes = max_field_bytes
        self._expose_errors = expose_errors
        self._text: str | None = None  # the open text message
        self._reasoning: str | None = None  # the open reasoning message
        self._step: str | None = None  # the agent whose step is open
        self._streamed = False  # the current model step streamed text
        self._parent: str | None = None  # the assistant message of the current model step, for its tool calls
        self._tools: set[str] = set()  # tool calls announced: a human-input wait replays the tool step

    def start(self, run_id: str) -> Iterable[Frame]:
        if self._run_id is None:
            self._run_id = run_id
        return [self._frame(self._ag.RunStartedEvent(thread_id=self._thread_id, run_id=self._run_id))]

    def encode(self, event: Event) -> Iterable[Frame]:
        ag = self._ag
        if isinstance(event, StopEvent):
            return self._terminal(event)
        if isinstance(event, ag.BaseEvent):
            return [self._frame(event)]  # AGUIChatWorkflow writes AG-UI events itself
        agents = agent_events()
        if agents is not None:
            if isinstance(event, agents.AgentInput):
                return self._model_step(event.current_agent_name)
            if isinstance(event, agents.AgentSetup):
                return []
            if isinstance(event, agents.AgentStream):
                return self._stream(event)
            if isinstance(event, agents.AgentOutput):
                return self._output(event)
            if isinstance(event, agents.ToolCallResult):
                return self._result(event)
            if isinstance(event, agents.ToolCall):
                return self._call(event.tool_id, event.tool_name, tool_kwargs(event))
            if isinstance(event, agents.AgentStreamStructuredOutput):
                return [self._custom("structured_output", event.output)]
        if isinstance(event, InputRequiredEvent):
            return [self._custom("input_required", fields(event))]
        if self._custom_events:
            return [self._custom(event_name(event), fields(event))]
        return []

    def abort(self, error: BaseException) -> Iterable[Frame]:
        frames = self._close_messages()
        if isinstance(error, asyncio.CancelledError):
            return [*frames, *self._finished(cancelled=True)]
        message = str(error) if self._expose_errors and str(error) else MASKED_ERROR
        return [*frames, self._frame(self._ag.RunErrorEvent(message=message, code="aborted"))]

    def fit(self, frame: Frame, fits: Fits) -> Iterable[Frame]:
        try:
            value: Any = json.loads(frame[1])
        except ValueError:
            return []
        if not isinstance(value, dict):
            return []
        event: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
        kind = event.get("type")
        delta = event.get("delta")
        if kind in _DELTAS and isinstance(delta, str):
            return split_text(delta, lambda piece: (frame[0], dumps({**event, "delta": piece})), fits) or []
        if kind in _CUTTABLE:
            fitted = shrink_strings(event, lambda shrunk: (frame[0], dumps(shrunk)), fits, _ELLIPSIS, _KEEP)
            return [] if fitted is None else [fitted]
        return []

    @staticmethod
    def _frame(event: BaseEvent) -> Frame:
        # The JSON ag_ui's EventEncoder writes after "data: ".
        return (None, event.model_dump_json(by_alias=True))

    def _custom(self, name: str, value: Any) -> Frame:
        bounded, _ = bounded_json(value, self._max_field_bytes, _ELLIPSIS)
        return self._frame(self._ag.CustomEvent(name=name, value=bounded))

    def _model_step(self, agent: str) -> list[Frame]:
        frames = self._close_messages()
        self._streamed = False
        self._parent = None
        if self._emit_steps and agent != self._step:  # the first step, or a hand-off
            frames += self._finish_step()
            self._step = agent
            frames.append(self._frame(self._ag.StepStartedEvent(step_name=agent)))
        return frames

    def _stream(self, event: AgentStream) -> list[Frame]:
        ag = self._ag
        frames: list[Frame] = []
        if event.thinking_delta:
            if self._reasoning is None:
                self._reasoning = str(uuid.uuid4())
                frames.append(self._frame(ag.ReasoningStartEvent(message_id=self._reasoning)))
                frames.append(self._frame(ag.ReasoningMessageStartEvent(message_id=self._reasoning, role="reasoning")))
            frames.append(
                self._frame(ag.ReasoningMessageContentEvent(message_id=self._reasoning, delta=event.thinking_delta))
            )
        if event.delta:
            frames += self._close_reasoning()
            if self._text is None:
                self._text = self._parent = str(uuid.uuid4())
                frames.append(self._frame(ag.TextMessageStartEvent(message_id=self._text, role="assistant")))
            self._streamed = True
            frames.append(self._frame(ag.TextMessageContentEvent(message_id=self._text, delta=event.delta)))
        return frames

    def _output(self, event: AgentOutput) -> list[Frame]:
        frames = self._close_messages()
        content = event.response.content or ""
        if not self._streamed and content:  # an agent built with streaming=False
            ag = self._ag
            self._parent = message_id = str(uuid.uuid4())
            frames.append(self._frame(ag.TextMessageStartEvent(message_id=message_id, role="assistant")))
            frames.append(self._frame(ag.TextMessageContentEvent(message_id=message_id, delta=content)))
            frames.append(self._frame(ag.TextMessageEndEvent(message_id=message_id)))
        return frames

    def _call(self, tool_id: str, tool_name: str, kwargs: dict[str, Any]) -> list[Frame]:
        if tool_id in self._tools:
            return []
        self._tools.add(tool_id)
        ag = self._ag
        arguments, _ = bounded_json(kwargs, self._max_field_bytes, _ELLIPSIS)
        start = ag.ToolCallStartEvent(tool_call_id=tool_id, tool_call_name=tool_name, parent_message_id=self._parent)
        return [
            *self._close_messages(),
            self._frame(start),
            self._frame(ag.ToolCallArgsEvent(tool_call_id=tool_id, delta=dumps(arguments))),
            self._frame(ag.ToolCallEndEvent(tool_call_id=tool_id)),
        ]

    def _result(self, event: ToolCallResult) -> list[Frame]:
        output = event.tool_output
        frames = self._call(event.tool_id, event.tool_name, tool_kwargs(event))  # nothing when announced
        if output.is_error and not self._expose_errors:
            content = MASKED_ERROR
        else:
            text = str(output.content)
            content = cut(text, self._max_field_bytes)
            content += _ELLIPSIS if content != text else ""
        result = self._ag.ToolCallResultEvent(
            message_id=str(uuid.uuid4()), tool_call_id=event.tool_id, content=content, role="tool"
        )
        frames.append(self._frame(result))
        return frames

    def _terminal(self, event: StopEvent) -> list[Frame]:
        ag = self._ag
        frames = self._close_messages()
        if isinstance(event, WorkflowFailedEvent):
            _, text = failure(event)
            message = text if self._expose_errors and text else MASKED_ERROR
            return [*frames, self._frame(ag.RunErrorEvent(message=message, code="failed"))]
        if isinstance(event, WorkflowTimedOutEvent):
            return [*frames, self._frame(ag.RunErrorEvent(message=TIMED_OUT, code="timeout"))]
        return [*frames, *self._finished(cancelled=isinstance(event, WorkflowCancelledEvent))]

    def _finished(self, *, cancelled: bool) -> list[Frame]:
        ag = self._ag
        outcome = ag.RunFinishedCancelledOutcome() if cancelled else None
        finished = ag.RunFinishedEvent(thread_id=self._thread_id, run_id=self._run_id or "", outcome=outcome)
        return [*self._finish_step(), self._frame(finished)]

    def _finish_step(self) -> list[Frame]:
        if self._step is None:
            return []
        step, self._step = self._step, None
        return [self._frame(self._ag.StepFinishedEvent(step_name=step))]

    def _close_reasoning(self) -> list[Frame]:
        if self._reasoning is None:
            return []
        ag = self._ag
        reasoning, self._reasoning = self._reasoning, None
        return [
            self._frame(ag.ReasoningMessageEndEvent(message_id=reasoning)),
            self._frame(ag.ReasoningEndEvent(message_id=reasoning)),
        ]

    def _close_messages(self) -> list[Frame]:
        frames = self._close_reasoning()
        if self._text is not None:
            frames.append(self._frame(self._ag.TextMessageEndEvent(message_id=self._text)))
            self._text = None
        return frames
