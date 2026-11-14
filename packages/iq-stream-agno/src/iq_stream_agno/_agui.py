"""AG-UI events as frames. The conversion is Agno's own (``agno.os.interfaces.agui.handlers``), one event at a time
so that frames go out between Agno's events; this module adds ``RUN_STARTED``, decides the terminal event from the
root run, and splits or cuts an event that does not fit the line limit."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, cast

from ag_ui.core import BaseEvent, EventType, RunFinishedEvent, RunStartedEvent
from agno.os.interfaces.agui import handlers as _handlers
from agno.os.interfaces.agui.handlers import is_completion_event, process_completion, process_event
from agno.os.interfaces.agui.state import StreamState
from agno.run.agent import RunCompletedEvent, RunErrorEvent
from agno.run.base import BaseRunOutputEvent
from iq_stream import FrameTooLarge

from ._common import MASKED_ERROR, ROOT_CANCELS, ROOT_ERRORS, Fits, Frame, Report, event_name, longest_prefix

try:  # ag-ui-protocol 1.0; agno's floor, 0.1.15, may not have it
    from ag_ui.core import RunFinishedCancelledOutcome as _Cancelled
except ImportError:  # pragma: no cover - depends on the installed ag-ui-protocol
    _Cancelled = None

#: Deltas a client concatenates, so an oversize one may be split into several events of its type.
_SPLITTABLE = frozenset({EventType.TEXT_MESSAGE_CONTENT, EventType.TOOL_CALL_ARGS, EventType.REASONING_MESSAGE_CONTENT})
_CUT = " [truncated]"
_ERROR_TEXT = {"RunError": "content", "TeamRunError": "content"}
_TOOL_EVENTS = frozenset({"ToolCallCompleted", "TeamToolCallCompleted", "ToolCallError", "TeamToolCallError"})


def dump(event: BaseEvent) -> str:
    """What ``ag_ui.encoder.EventEncoder`` writes after ``data:``."""
    return event.model_dump_json(by_alias=True, exclude_none=True)


def cancelled(event: BaseEvent) -> BaseEvent:
    """``RUN_FINISHED`` with the cancelled outcome, so a browser reads from the data layer that the ``iq:fail``
    after it was a cancel. Unchanged where the installed ag-ui-protocol has no such outcome."""
    if _Cancelled is None or event.type != EventType.RUN_FINISHED:
        return event
    return event.model_copy(update={"outcome": _Cancelled()})


class AguiFramer:
    """AG-UI events as frames, under the line limit: an oversize delta is split, a tool result or an error message
    cut with a marker, anything else dropped and reported."""

    def __init__(
        self,
        *,
        event_names: bool,
        mask_errors: bool,
        shrink: bool,
        fits: Fits,
        size: Callable[[str, str | None], int],
        limit: int,
        report: Report,
    ) -> None:
        self._event_names = event_names
        self._mask_errors = mask_errors
        self._shrink = shrink
        self._fits = fits
        self._size = size
        self._limit = limit
        self._report = report

    def frames(self, events: list[BaseEvent]) -> list[Frame]:
        return [frame for event in events for frame in self._frames(event)]

    def _frames(self, event: BaseEvent) -> list[Frame]:
        if self._mask_errors and event.type == EventType.RUN_ERROR:
            event = event.model_copy(update={"message": MASKED_ERROR, "raw_event": None})
        name = str(event.type.value) if self._event_names else None
        data = dump(event)
        if self._fits(data, name):
            return [(name, data)]
        size = self._size(data, name)
        frames = self._fitted(event, name) if self._shrink else None
        if frames is None:
            self._report(FrameTooLarge(size, self._limit))
            return []
        return frames

    def _fitted(self, event: BaseEvent, name: str | None) -> list[Frame] | None:
        def fits(update: Mapping[str, object]) -> bool:
            return self._fits(dump(event.model_copy(update=dict(update))), name)

        if event.type in _SPLITTABLE:
            rest: str = getattr(event, "delta", "")
            pieces: list[Frame] = []
            while rest:
                length = longest_prefix(rest, lambda prefix: fits({"delta": prefix}))
                if length <= 0:
                    return None
                pieces.append((name, dump(event.model_copy(update={"delta": rest[:length]}))))
                rest = rest[length:]
            return pieces
        if event.type == EventType.RUN_ERROR:
            event = event.model_copy(update={"raw_event": None})
            if fits({}):
                return [(name, dump(event))]
            return self._cut(event, "message", fits, name)
        if event.type == EventType.TOOL_CALL_RESULT:
            return self._cut(event, "content", fits, name)
        return None

    def _cut(
        self, event: BaseEvent, field: str, fits: Callable[[Mapping[str, object]], bool], name: str | None
    ) -> list[Frame] | None:
        text = str(getattr(event, field, ""))
        length = longest_prefix(text, lambda prefix: fits({field: prefix + _CUT}))
        if length < 0:
            return None
        return [(name, dump(event.model_copy(update={field: text[:length] + _CUT})))]


def _masked(event: BaseRunOutputEvent) -> BaseRunOutputEvent:
    """A copy of an error event, or of a failed tool's event, without the error's text."""
    name = event_name(event)
    field = _ERROR_TEXT.get(name)
    tool: object = getattr(event, "tool", None)
    failed_tool = name in _TOOL_EVENTS and getattr(tool, "tool_call_error", False) and getattr(tool, "result", None)
    if field is None and not failed_tool:
        return event
    event = copy.copy(event)
    if field is not None:
        setattr(event, field, MASKED_ERROR)
        if hasattr(event, "additional_data"):
            setattr(event, "additional_data", None)  # noqa: B010 - the error events have it, the others do not
    if failed_tool and dataclasses.is_dataclass(tool) and not isinstance(tool, type):
        setattr(event, "tool", dataclasses.replace(tool, result=MASKED_ERROR))  # noqa: B010
    return event


class AguiEncoder:
    """One Agno run as AG-UI events, through Agno's converter. Only the root run's terminal event ends the AG-UI run:
    a team member's or a workflow step's ``RunCompleted`` or ``RunError`` is passed on as ``RAW``."""

    def __init__(self, framer: AguiFramer, *, expose_errors: bool) -> None:
        self._framer = framer
        self._expose_errors = expose_errors
        self._state: StreamState | None = None
        self._terminal: BaseRunOutputEvent | None = None

    def _start(self, thread_id: str | None, run_id: str | None) -> tuple[StreamState, list[BaseEvent]]:
        if self._state is not None:
            return self._state, []
        self._state = StreamState(thread_id=thread_id or "", run_id=run_id or "")
        started = RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id or "", run_id=run_id or "")
        return self._state, [started]

    def encode(
        self, event: BaseRunOutputEvent, *, root: bool, thread_id: str | None, run_id: str | None
    ) -> list[Frame]:
        state, events = self._start(thread_id, run_id)
        if not self._expose_errors:
            event = _masked(event)
        if root and is_completion_event(event):
            self._terminal = event  # held, as Agno's own stream does, and converted when the run has ended
        else:
            events.extend(process_event(event, state))
        return self._framer.frames(events)

    def finish(
        self,
        status: str,
        *,
        error: BaseException | None,
        root_error: BaseRunOutputEvent | None,
        thread_id: str | None,
        run_id: str | None,
    ) -> list[Frame]:
        state, events = self._start(thread_id, run_id)
        terminal = self._terminal
        if status == "error":
            if terminal is None or event_name(terminal) not in _ERROR_TEXT:
                terminal = self._error_event(error, root_error, thread_id, run_id)
            if hasattr(_handlers, "on_run_error"):
                events.extend(process_completion(terminal, state))
            else:  # agno 2.x converts a RunError to RAW and finishes the run: a failed run must not read as finished
                events.extend(_run_error(terminal, state))
        else:
            finished = process_completion(terminal or RunCompletedEvent(), state)
            if status in ("cancelled", "interrupted"):
                finished = [cancelled(event) for event in finished]
            events.extend(finished)
        return self._framer.frames(events)

    def _error_event(
        self,
        error: BaseException | None,
        root_error: BaseRunOutputEvent | None,
        thread_id: str | None,
        run_id: str | None,
    ) -> BaseRunOutputEvent:
        """An Agno ``RunError`` for an ending Agno did not describe as one: a ``WorkflowError``, an iterator that
        raised, a ``before_close`` that failed. Agno's converter then closes what is open and emits ``RUN_ERROR``."""
        code: str | None = None
        if root_error is not None:
            text: object = getattr(root_error, "error", None) or getattr(root_error, "content", None)
            raw_code: object = getattr(root_error, "error_type", None)
            code = raw_code if isinstance(raw_code, str) else None
        elif error is not None:
            text = str(error) or type(error).__name__
        else:
            text = None
        message = str(text) if text and self._expose_errors else MASKED_ERROR
        return RunErrorEvent(run_id=run_id, session_id=thread_id, content=message, error_type=code)


def _run_error(terminal: BaseRunOutputEvent, state: StreamState) -> list[BaseEvent]:
    """What agno 3's converter emits for a ``RunError``: the open spans closed, then ``RUN_ERROR``."""
    from ag_ui.core import (
        ReasoningEndEvent,
        ReasoningMessageEndEvent,
        TextMessageEndEvent,
        ToolCallEndEvent,
    )
    from ag_ui.core import (
        RunErrorEvent as AguiRunError,
    )

    events: list[BaseEvent] = []
    if state.reasoning_message_id is not None:
        events.append(ReasoningMessageEndEvent(message_id=state.reasoning_message_id))
        events.append(ReasoningEndEvent(message_id=state.reasoning_message_id))
        state.end_reasoning()
    for tool_call_id in sorted(state.active_tool_call_ids - state.ended_tool_call_ids):
        events.append(ToolCallEndEvent(tool_call_id=tool_call_id))
        state.end_tool_call(tool_call_id)
    if state.text_message_open:
        events.append(TextMessageEndEvent(message_id=state.text_message_id))
        state.close_text_message()
    code: object = getattr(terminal, "error_type", None)
    message = str(getattr(terminal, "content", None) or MASKED_ERROR)
    events.append(AguiRunError(message=message, code=code if isinstance(code, str) else None))
    return events


class AguiTrack:
    """The terminal decision for a stream of AG-UI events that Agno's ``run_entity()`` produced (the AgentOS
    interface's AG-UI route): ``RUN_ERROR`` fails the run, a ``RAW`` cancelled event of the root run cancels it. A
    stream that stops without a terminal event gets one: what is open is closed, and the run ends with ``RUN_ERROR``
    or, when it was interrupted, ``RUN_FINISHED`` with the cancelled outcome."""

    def __init__(self, framer: AguiFramer, *, run_id: str, thread_id: str) -> None:
        self._framer = framer
        self.run_id: str | None = run_id
        self.session_id: str | None = thread_id
        self._failed = False
        self._cancelled = False
        self._raw_error: dict[str, Any] | None = None
        self._terminated = False
        self._started = False
        self._open: dict[str, list[str]] = {"message": [], "tool": [], "reasoning": []}

    def frames(self, item: object, encode: bool) -> list[Frame]:
        if not isinstance(item, BaseEvent):
            return []
        self._observe(item)
        if not encode:
            return []
        if item.type == EventType.RUN_FINISHED and self._raw_error is not None:
            # agno 2.x converts the root run's RunError to RAW and then finishes the run.
            from ag_ui.core import RunErrorEvent

            self._failed = True
            code: object = self._raw_error.get("error_type")
            message = str(self._raw_error.get("content") or MASKED_ERROR)
            item = RunErrorEvent(message=message, code=code if isinstance(code, str) else None)
        elif item.type == EventType.RUN_FINISHED and self._cancelled:
            item = cancelled(item)
        return self._framer.frames([item])

    def status(self) -> str:
        return "error" if self._failed else "cancelled" if self._cancelled else "completed"

    def final_frames(self, status: str, error: BaseException | None) -> list[Frame]:
        if self._terminated:
            return []
        from ag_ui.core import ReasoningMessageEndEvent, RunErrorEvent, TextMessageEndEvent, ToolCallEndEvent

        events: list[BaseEvent] = []
        if not self._started:
            events.append(
                RunStartedEvent(type=EventType.RUN_STARTED, thread_id=self.session_id or "", run_id=self.run_id or "")
            )
        events.extend(ReasoningMessageEndEvent(message_id=item) for item in self._open["reasoning"])
        events.extend(ToolCallEndEvent(tool_call_id=item) for item in self._open["tool"])
        events.extend(TextMessageEndEvent(message_id=item) for item in self._open["message"])
        if status == "error":
            message = str(error) if error is not None and str(error) else MASKED_ERROR
            events.append(RunErrorEvent(message=message))
        else:
            finished = RunFinishedEvent(thread_id=self.session_id or "", run_id=self.run_id or "")
            events.append(cancelled(finished) if status in ("cancelled", "interrupted") else finished)
        self._terminated = True
        return self._framer.frames(events)

    def _observe(self, event: BaseEvent) -> None:
        kind = event.type
        if kind == EventType.RUN_STARTED:
            self._started = True
        elif kind == EventType.RUN_ERROR:
            self._failed = self._terminated = True
        elif kind == EventType.RUN_FINISHED:
            self._terminated = True
        elif kind == EventType.RAW:
            raw: object = getattr(event, "event", None)
            if isinstance(raw, dict):
                raw = cast("dict[str, Any]", raw)
                if raw.get("run_id") == self.run_id and raw.get("event") in ROOT_CANCELS:
                    self._cancelled = True
                elif raw.get("run_id") == self.run_id and raw.get("event") in ROOT_ERRORS:
                    self._raw_error = raw
        span = _SPANS.get(kind)
        if span is not None:
            group, opening, key = span
            value: object = getattr(event, key, None)
            if isinstance(value, str):
                if opening:
                    self._open[group].append(value)
                elif value in self._open[group]:
                    self._open[group].remove(value)


_SPANS = {
    EventType.TEXT_MESSAGE_START: ("message", True, "message_id"),
    EventType.TEXT_MESSAGE_END: ("message", False, "message_id"),
    EventType.TOOL_CALL_START: ("tool", True, "tool_call_id"),
    EventType.TOOL_CALL_END: ("tool", False, "tool_call_id"),
    EventType.REASONING_MESSAGE_START: ("reasoning", True, "message_id"),
    EventType.REASONING_MESSAGE_END: ("reasoning", False, "message_id"),
}
