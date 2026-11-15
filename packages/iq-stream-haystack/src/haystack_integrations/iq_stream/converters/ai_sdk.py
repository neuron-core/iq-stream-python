"""StreamingChunk -> AI SDK UI message chunks, the stream that ``useChat`` reads."""

from __future__ import annotations

import json
from typing import Unpack

from haystack.dataclasses import ToolCallResult

from ._messages import Call, MessageConverter
from .base import INVALID_ARGUMENTS, ConverterOptions, Frame, dumps

# Haystack's finish reasons, in the AI SDK's spelling.
_FINISH_REASONS = {"stop": "stop", "length": "length", "content_filter": "content-filter", "tool_calls": "tool-calls"}


class AISDKConverter(MessageConverter):
    """One UI message chunk per frame, unnamed: ``start``, then a step per model call and its tool results
    (``start-step`` ... ``finish-step``), then ``finish``, ``error`` or, for a cancelled run, ``abort``.

    The whole run is one assistant message, whose id is ``message_id``. Deltas too large for a frame are split; a
    tool's output is truncated with a marker; a ``tool-input-available`` too large for a frame is left out, since
    the input deltas carried the arguments already. A call whose arguments are not JSON, cut off by a broken
    stream or by the length limit, ends with ``tool-input-error``: Haystack runs no such call.
    """

    def __init__(self, *, message_id: str, **options: Unpack[ConverterOptions]) -> None:
        super().__init__(**options)
        self.message_id = message_id
        self._step_open = False
        self._waiting: set[str] = set()  # tool calls of the open step that have no result yet
        self._last_reason: str | None = None

    def start(self) -> list[Frame]:
        return [Frame(self.fit.whole(dumps({"type": "start", "messageId": self.message_id})))]

    def finish(self) -> list[Frame]:
        chunk: dict[str, str] = {"type": "finish"}
        if self._last_reason in _FINISH_REASONS:
            chunk["finishReason"] = _FINISH_REASONS[self._last_reason]
        return self._frames([*self._close_all(), *self._finish_step(), dumps(chunk)])

    def cancelled(self) -> list[Frame]:
        return self._frames([*self._close_all(), *self._finish_step(), dumps({"type": "abort", "reason": "cancelled"})])

    def error(self, exc: BaseException) -> list[Frame]:
        def build(text: str) -> str:
            return dumps({"type": "error", "errorText": text})

        text = self.fit.truncate(self._error_message(exc), build)
        return self._frames([*self._close_all(), *self._finish_step(), build(text)])

    def custom(self, name: str, value: object) -> list[Frame]:
        return [Frame(self.fit.whole(dumps({"type": f"data-{name}", "data": value})))]

    def _message_opened(self) -> list[str]:
        if self._step_open:
            return []
        self._step_open = True
        return [dumps({"type": "start-step"})]

    def _message_finished(self, reason: str | None) -> list[str]:
        self._last_reason = reason
        return [] if self._waiting else self._finish_step()

    def _results_finished(self) -> list[str]:
        self._waiting.clear()
        return self._finish_step()

    def _finish_step(self) -> list[str]:
        # finish-step forgets the open text and reasoning parts: it waits until no source has a message open.
        if not self._step_open or self._messages:
            return []
        self._step_open = False
        return [dumps({"type": "finish-step"})]

    def _start_reasoning(self, reasoning_id: str) -> list[str]:
        return [dumps({"type": "reasoning-start", "id": reasoning_id})]

    def _reasoning_delta(self, reasoning_id: str, delta: str) -> list[str]:
        def build(text: str) -> str:
            return dumps({"type": "reasoning-delta", "id": reasoning_id, "delta": text})

        return [build(piece) for piece in self.fit.split(delta, build)]

    def _end_reasoning(self, reasoning_id: str) -> list[str]:
        return [dumps({"type": "reasoning-end", "id": reasoning_id})]

    def _start_text(self, text_id: str) -> list[str]:
        return [dumps({"type": "text-start", "id": text_id})]

    def _text_delta(self, text_id: str, delta: str) -> list[str]:
        def build(text: str) -> str:
            return dumps({"type": "text-delta", "id": text_id, "delta": text})

        return [build(piece) for piece in self.fit.split(delta, build)]

    def _end_text(self, text_id: str) -> list[str]:
        return [dumps({"type": "text-end", "id": text_id})]

    def _start_call(self, call: Call, parent_id: str) -> list[str]:
        return [self.fit.whole(dumps({"type": "tool-input-start", "toolCallId": call.id, "toolName": call.name}))]

    def _call_delta(self, call: Call, delta: str) -> list[str]:
        def build(text: str) -> str:
            return dumps({"type": "tool-input-delta", "toolCallId": call.id, "inputTextDelta": text})

        return [build(piece) for piece in self.fit.split(delta, build)]

    def _end_calls(self, calls: list[Call]) -> list[str]:
        out: list[str] = []
        for call in calls:
            try:
                arguments: object = json.loads(call.text or "{}")
            except ValueError:
                out.append(self._input_error(call))
                continue
            self._waiting.add(call.id)
            chunk = dumps(
                {"type": "tool-input-available", "toolCallId": call.id, "toolName": call.name, "input": arguments}
            )
            if self.fit.fits(chunk):
                out.append(chunk)
        return out

    def _input_error(self, call: Call) -> str:
        def build(text: str) -> str:
            return dumps(
                {
                    "type": "tool-input-error",
                    "toolCallId": call.id,
                    "toolName": call.name,
                    "input": text,
                    "errorText": INVALID_ARGUMENTS,
                }
            )

        chunk = build(call.text)
        return chunk if self.fit.fits(chunk) else self.fit.whole(build(""))  # the input deltas carried it

    def _replay_call(self, call: Call) -> list[str]:
        return [*self._start_call(call, ""), *self._end_calls([call])]

    def _result_frames(self, call_id: str, result: ToolCallResult) -> list[str]:
        self._waiting.discard(call_id)
        content = self._result_content(result)
        if result.error:

            def build(text: str) -> str:
                return dumps({"type": "tool-output-error", "toolCallId": call_id, "errorText": text})

        else:

            def build(text: str) -> str:
                return dumps({"type": "tool-output-available", "toolCallId": call_id, "output": text})

        return [build(self.fit.truncate(content, build, self.max_tool_result_bytes))]
