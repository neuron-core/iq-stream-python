"""StreamingChunk -> its own JSON: one frame per chunk, for a page that reads Haystack's chunks as they are."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import replace
from typing import Any

from haystack.dataclasses import StreamingChunk, ToolCallDelta
from iq_stream import FrameTooLarge

from .base import BaseConverter, Frame, dumps


class NativeConverter(BaseConverter):
    """Each chunk as ``{"type": "chunk", ...}`` with the fields of ``StreamingChunk.to_dict()`` but ``meta`` (which
    may hold objects that are not JSON) and the providers' ``extra``; a tool result is its text. The run opens with
    ``{"type": "start"}`` and ends with ``{"type": "finish"}``, ``{"type": "error", "message": ...}`` or
    ``{"type": "cancelled"}``.

    A chunk too large for a frame is split into several chunks of the same kind (its ``start`` on the first, its
    ``finish_reason`` on the last); a tool result is truncated with a marker that gives its full length.
    """

    def start(self) -> list[Frame]:
        return [Frame(dumps({"type": "start"}))]

    def finish(self) -> list[Frame]:
        return [Frame(dumps({"type": "finish"}))]

    def cancelled(self) -> list[Frame]:
        return [Frame(dumps({"type": "cancelled"}))]

    def error(self, exc: BaseException) -> list[Frame]:
        def build(text: str) -> str:
            return dumps({"type": "error", "message": text})

        return [Frame(build(self.fit.truncate(self._error_message(exc), build)))]

    def custom(self, name: str, value: object) -> list[Frame]:
        return [Frame(self.fit.whole(dumps({"type": "custom", "name": name, "value": value})))]

    def convert(self, chunk: StreamingChunk, source: Hashable) -> list[Frame]:
        data = self._encode(chunk)
        if self.fit.fits(data):
            return [Frame(data)]
        if chunk.content:
            pieces = self._split(chunk, lambda part: replace(chunk, content=part), chunk.content)
        elif chunk.reasoning is not None and chunk.reasoning.reasoning_text:
            reasoning = chunk.reasoning
            pieces = self._split(
                chunk,
                lambda part: replace(chunk, reasoning=replace(reasoning, reasoning_text=part)),
                reasoning.reasoning_text,
            )
        elif chunk.tool_calls:
            pieces = [piece for delta in chunk.tool_calls for piece in self._split_call(chunk, delta)]
        else:  # a tool result is truncated to fit already
            raise FrameTooLarge(self.fit.measure(data), self.fit.limit)
        return self._frames(self._encode(piece) for piece in pieces)

    def _encode(self, chunk: StreamingChunk, result_text: str | None = None) -> str:
        fields: dict[str, Any] = {
            "type": "chunk",
            "content": chunk.content,
            "component_info": chunk.component_info.to_dict() if chunk.component_info else None,
            "index": chunk.index,
            "tool_calls": [
                {"index": delta.index, "tool_name": delta.tool_name, "arguments": delta.arguments, "id": delta.id}
                for delta in chunk.tool_calls
            ]
            if chunk.tool_calls
            else None,
            "tool_call_result": None,
            "start": chunk.start,
            "finish_reason": chunk.finish_reason,
            "reasoning": {"reasoning_text": chunk.reasoning.reasoning_text} if chunk.reasoning else None,
        }
        result = chunk.tool_call_result
        if result is not None:
            if result_text is None:
                result_text = self.fit.truncate(
                    self._result_content(result), lambda text: self._encode(chunk, text), self.max_tool_result_bytes
                )
            origin = result.origin.to_dict()
            fields["tool_call_result"] = {"result": result_text, "origin": origin, "error": result.error}
        return dumps(fields)

    def _split(
        self, chunk: StreamingChunk, with_text: Callable[[str], StreamingChunk], text: str
    ) -> list[StreamingChunk]:
        # Measured with start=False, the longer spelling, and with the finish_reason: the worst case of every piece.
        parts = self.fit.split(text, lambda part: self._encode(replace(with_text(part), start=False)))
        last = len(parts) - 1
        return [
            replace(
                with_text(part),
                start=chunk.start and number == 0,
                finish_reason=chunk.finish_reason if number == last else None,
            )
            for number, part in enumerate(parts)
        ]

    def _split_call(self, chunk: StreamingChunk, delta: ToolCallDelta) -> list[StreamingChunk]:
        single = replace(chunk, tool_calls=[delta])
        if not delta.arguments or self.fit.fits(self._encode(single)):
            return [single]
        pieces = self._split(
            chunk, lambda part: replace(chunk, tool_calls=[replace(delta, arguments=part)]), delta.arguments
        )
        # Only the first fragment keeps the call's id and name, as providers send them.
        return [
            replace(piece, tool_calls=[replace(delta, id=None, tool_name=None, arguments=self._arguments(piece))])
            if number
            else piece
            for number, piece in enumerate(pieces)
        ]

    @staticmethod
    def _arguments(chunk: StreamingChunk) -> str | None:
        return chunk.tool_calls[0].arguments if chunk.tool_calls else None
