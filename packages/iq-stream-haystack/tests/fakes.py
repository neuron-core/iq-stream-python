"""Scripted chat generators that stream the way providers do, without a model, a key or a network.

A script is one ``Turn`` per model call: the chunks the generator streams and the reply it returns. The turn is
picked by the number of assistant messages in the conversation, so one generator serves any number of runs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from haystack import component
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    ToolCall,
    select_streaming_callback,
)
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from haystack.tools import ToolsType

Shape = Literal["openai", "mock", "anthropic"]


@dataclass
class Turn:
    chunks: list[StreamingChunk]
    reply: ChatMessage
    #: Raise this after streaming ``fail_after`` chunks, as a provider whose stream breaks.
    error: Exception | None = None
    fail_after: int = 0
    meta: dict[str, Any] = field(default_factory=lambda: {})


def text_turn(words: Sequence[str], *, reasoning: Sequence[str] = (), finish: str | None = "stop") -> Turn:
    """A text answer, word by word, after optional reasoning deltas."""
    chunks: list[StreamingChunk] = []
    for number, delta in enumerate(reasoning):
        chunks.append(StreamingChunk(content="", index=0, start=number == 0, reasoning=ReasoningContent(delta)))
    block = 1 if reasoning else 0
    for number, word in enumerate(words):
        chunks.append(StreamingChunk(content=word, index=block, start=number == 0))
    if finish is not None:
        chunks.append(StreamingChunk(content="", finish_reason=finish))  # pyright: ignore[reportArgumentType]
    text = "".join(words)
    return Turn(chunks, ChatMessage.from_assistant(text, reasoning="".join(reasoning) or None))


def tool_turn(
    calls: Sequence[ToolCall],
    *,
    text: Sequence[str] = (),
    reasoning: Sequence[str] = (),
    shape: Shape = "openai",
    finish: bool = True,
) -> Turn:
    """Tool calls, after optional reasoning and text.

    ``openai``: a start delta with the id and name, then the arguments in two fragments. ``mock``: one delta per call
    with the complete arguments, as Haystack's MockChatGenerator sends them. ``anthropic``: an empty reasoning chunk
    when the thinking block opens and one carrying only a signature, then the calls as fragments.
    """
    chunks: list[StreamingChunk] = []
    block = 0
    if shape == "anthropic":
        chunks.append(StreamingChunk(content="", index=0, start=True, reasoning=ReasoningContent("")))
    for number, delta in enumerate(reasoning):
        start = number == 0 and shape != "anthropic"
        chunks.append(StreamingChunk(content="", index=0, start=start, reasoning=ReasoningContent(delta)))
    if shape == "anthropic":
        signature = ReasoningContent("", extra={"signature": "secret-signature"})
        chunks.append(StreamingChunk(content="", index=0, reasoning=signature))
    if reasoning or shape == "anthropic":
        block += 1
    for number, word in enumerate(text):
        chunks.append(StreamingChunk(content=word, index=block, start=number == 0))
    if text:
        block += 1
    for call in calls:
        arguments = json.dumps(call.arguments)
        if shape == "mock":
            delta = ToolCallDelta(index=block, tool_name=call.tool_name, arguments=arguments, id=call.id)
            chunks.append(StreamingChunk(content="", index=block, start=True, tool_calls=[delta]))
        else:
            opening = ToolCallDelta(
                index=block, tool_name=call.tool_name, id=call.id, extra={"provider": "secret-extra"}
            )
            chunks.append(StreamingChunk(content="", index=block, start=True, tool_calls=[opening]))
            for fragment in (arguments[:8], arguments[8:]):
                delta = ToolCallDelta(index=block, arguments=fragment)
                chunks.append(StreamingChunk(content="", index=block, tool_calls=[delta], meta={"raw": object()}))
        block += 1
    if finish:
        chunks.append(StreamingChunk(content="", finish_reason="tool_calls"))
    reply = ChatMessage.from_assistant(
        "".join(text) or None, tool_calls=list(calls), reasoning="".join(reasoning) or None
    )
    return Turn(chunks, reply)


class _Scripted:
    """What the scripted generators share; the component classes add ``run`` and ``run_async``."""

    def __init__(self, turns: Sequence[Turn], delay: float = 0.0) -> None:
        self.turns = list(turns)
        self.delay = delay
        self.calls = 0

    def _turn(self, messages: list[ChatMessage]) -> Turn:
        self.calls += 1
        answered = sum(1 for message in messages if message.is_from("assistant"))
        return self.turns[min(answered, len(self.turns) - 1)]

    def _chunks(self, turn: Turn) -> list[StreamingChunk]:
        info = ComponentInfo.from_component(self)  # pyright: ignore[reportArgumentType]
        return [_with_info(chunk, info) for chunk in turn.chunks]

    def _stream_sync(self, messages: list[ChatMessage], callback: StreamingCallbackT | None) -> dict[str, Any]:
        turn = self._turn(messages)
        for number, chunk in enumerate(self._chunks(turn)):
            if turn.error is not None and number == turn.fail_after:
                raise turn.error
            if self.delay:
                time.sleep(self.delay)
            if callback is not None:
                callback(chunk)
        if turn.error is not None:
            raise turn.error
        return {"replies": [turn.reply]}


def _with_info(chunk: StreamingChunk, info: ComponentInfo) -> StreamingChunk:
    return StreamingChunk(
        content=chunk.content,
        meta=chunk.meta,
        component_info=info if chunk.tool_call_result is None else None,
        index=chunk.index,
        tool_calls=chunk.tool_calls,
        tool_call_result=chunk.tool_call_result,
        start=chunk.start,
        finish_reason=chunk.finish_reason,
        reasoning=chunk.reasoning,
    )


@component
class ScriptedChatGenerator(_Scripted):
    """Streams its script in ``run`` and in ``run_async``."""

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, Any]:
        callback = select_streaming_callback(None, streaming_callback, requires_async=False)
        return self._stream_sync(messages, callback)

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, Any]:
        callback = select_streaming_callback(None, streaming_callback, requires_async=True)
        turn = self._turn(messages)
        for number, chunk in enumerate(self._chunks(turn)):
            if turn.error is not None and number == turn.fail_after:
                raise turn.error
            await asyncio.sleep(self.delay)
            if callback is not None:
                result = callback(chunk)
                if inspect.isawaitable(result):
                    await result
        if turn.error is not None:
            raise turn.error
        return {"replies": [turn.reply]}


@component
class SyncOnlyChatGenerator(_Scripted):
    """No ``run_async``: Haystack runs it in a worker thread during an async run."""

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> dict[str, Any]:
        callback = select_streaming_callback(None, streaming_callback, requires_async=False)
        return self._stream_sync(messages, callback)


def weather_call(city: str = "Paris", call_id: str | None = "call_1") -> ToolCall:
    return ToolCall(tool_name="get_weather", arguments={"city": city}, id=call_id)


def weather_script(*, shape: Shape = "openai", city: str = "Paris") -> list[Turn]:
    """The run most tests use: some text, one tool call, then the answer."""
    return [
        tool_turn([weather_call(city)], text=["Let ", "me ", "check."], shape=shape),
        text_turn(["It ", "is ", "sunny ", "in ", f"{city}."]),
    ]


def result_chunk(call: ToolCall, result: Any = "Sunny, 21C", *, error: bool = False, index: int = 0) -> StreamingChunk:
    """A tool result as the Agent streams it, with the ToolCall object in ``meta`` as Haystack puts it there."""
    from haystack.dataclasses import ToolCallResult

    return StreamingChunk(
        content="",
        index=index,
        start=True,
        tool_call_result=ToolCallResult(result=result, origin=call, error=error),
        meta={"tool_result": result, "tool_call": call},
    )


RESULTS_DONE = StreamingChunk(
    content="", finish_reason="tool_call_results", meta={"finish_reason": "tool_call_results"}
)
