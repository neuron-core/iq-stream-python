"""StreamingChunk -> AG-UI events, built with the ``ag_ui.core`` models."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunErrorEvent,
    RunFinishedCancelledOutcome,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from haystack.dataclasses import ToolCallResult

from ._messages import Call, MessageConverter
from .base import ConverterOptions, Frame, dumps, new_id

if TYPE_CHECKING:
    from typing_extensions import Unpack


def encode_event(event: BaseEvent) -> str:
    """An event's JSON as the TypeScript client reads it: camelCase, unset fields left out. A value JSON cannot hold
    is written as its ``str()``, and a NaN or an infinity as ``null``, as ``dumps`` writes them."""
    return event.model_dump_json(by_alias=True, exclude_none=True, fallback=str)


class AGUIConverter(MessageConverter):
    """One AG-UI event per frame, unnamed: ``RUN_STARTED``, text, reasoning and tool call events, then
    ``RUN_FINISHED`` (with the ``cancelled`` outcome for a cancelled run) or ``RUN_ERROR``.

    Text, reasoning and argument deltas too large for a frame are split into several events; a tool result is
    truncated with a marker that gives its full length.
    """

    def __init__(self, *, thread_id: str, run_id: str, **options: Unpack[ConverterOptions]) -> None:
        super().__init__(**options)
        self.thread_id = thread_id
        self.run_id = run_id

    def start(self) -> list[Frame]:
        return [Frame(self.fit.whole(encode_event(RunStartedEvent(thread_id=self.thread_id, run_id=self.run_id))))]

    def finish(self) -> list[Frame]:
        event = RunFinishedEvent(thread_id=self.thread_id, run_id=self.run_id)
        return self._frames([*self._close_all(), self.fit.whole(encode_event(event))])

    def cancelled(self) -> list[Frame]:
        event = RunFinishedEvent(thread_id=self.thread_id, run_id=self.run_id, outcome=RunFinishedCancelledOutcome())
        return self._frames([*self._close_all(), self.fit.whole(encode_event(event))])

    def error(self, exc: BaseException) -> list[Frame]:
        message = self.fit.truncate(
            self._error_message(exc), lambda text: encode_event(RunErrorEvent(message=text, code="error"))
        )
        return self._frames([*self._close_all(), encode_event(RunErrorEvent(message=message, code="error"))])

    def custom(self, name: str, value: object) -> list[Frame]:
        # Through the JSON the other protocols write, so that the three write every value alike.
        event = CustomEvent(name=name, value=json.loads(dumps(value)))
        return [Frame(self.fit.whole(encode_event(event)))]

    def _start_reasoning(self, reasoning_id: str) -> list[str]:
        return [
            encode_event(ReasoningStartEvent(message_id=reasoning_id)),
            encode_event(ReasoningMessageStartEvent(message_id=reasoning_id, role="reasoning")),
        ]

    def _reasoning_delta(self, reasoning_id: str, delta: str) -> list[str]:
        def build(text: str) -> str:
            return encode_event(ReasoningMessageContentEvent(message_id=reasoning_id, delta=text))

        return [build(piece) for piece in self.fit.split(delta, build)]

    def _end_reasoning(self, reasoning_id: str) -> list[str]:
        return [
            encode_event(ReasoningMessageEndEvent(message_id=reasoning_id)),
            encode_event(ReasoningEndEvent(message_id=reasoning_id)),
        ]

    def _start_text(self, text_id: str) -> list[str]:
        return [encode_event(TextMessageStartEvent(message_id=text_id, role="assistant"))]

    def _text_delta(self, text_id: str, delta: str) -> list[str]:
        def build(text: str) -> str:
            return encode_event(TextMessageContentEvent(message_id=text_id, delta=text))

        return [build(piece) for piece in self.fit.split(delta, build)]

    def _end_text(self, text_id: str) -> list[str]:
        return [encode_event(TextMessageEndEvent(message_id=text_id))]

    def _start_call(self, call: Call, parent_id: str) -> list[str]:
        event = ToolCallStartEvent(tool_call_id=call.id, tool_call_name=call.name, parent_message_id=parent_id)
        return [self.fit.whole(encode_event(event))]

    def _call_delta(self, call: Call, delta: str) -> list[str]:
        def build(text: str) -> str:
            return encode_event(ToolCallArgsEvent(tool_call_id=call.id, delta=text))

        return [build(piece) for piece in self.fit.split(delta, build)]

    def _end_calls(self, calls: list[Call]) -> list[str]:
        return [encode_event(ToolCallEndEvent(tool_call_id=call.id)) for call in calls]

    def _replay_call(self, call: Call) -> list[str]:
        return [
            *self._start_call(call, new_id("msg")),
            *self._call_delta(call, call.text),
            *self._end_calls([call]),
        ]

    def _result_frames(self, call_id: str, result: ToolCallResult) -> list[str]:
        message_id = new_id("tool")

        def build(text: str) -> str:
            return encode_event(
                ToolCallResultEvent(message_id=message_id, tool_call_id=call_id, content=text, role="tool")
            )

        return [build(self.fit.truncate(self._result_content(result), build, self.max_tool_result_bytes))]
