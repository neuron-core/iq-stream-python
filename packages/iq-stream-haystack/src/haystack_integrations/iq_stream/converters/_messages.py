"""The bookkeeping that the AG-UI and the AI SDK converters share: which model messages are open, from which
source, with which text, reasoning and tool calls, and which tool call a result answers."""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal, Unpack

from haystack.dataclasses import StreamingChunk, ToolCall, ToolCallResult

from .base import MODEL_FINISH, BaseConverter, ConverterOptions, Frame, new_id

Block = Literal["reasoning", "text", "call"]


@dataclass
class Call:
    id: str
    name: str
    #: The id was made here, since the provider gave none.
    generated: bool
    arguments: list[str] = field(default_factory=lambda: [])

    @property
    def text(self) -> str:
        return "".join(self.arguments)


@dataclass
class Message:
    """The open parts of one model message from one source."""

    id: str
    #: The message a tool call belongs to: the last text block, or the message itself.
    parent_id: str
    text_id: str | None = None
    reasoning_id: str | None = None
    texts: int = 0
    calls: dict[int, Call] = field(default_factory=lambda: {})
    #: The block indexes that carried something, by kind, and the highest: a provider numbers a message's blocks
    #: upwards, so a block that starts again, or below the highest, belongs to another model call.
    blocks: dict[Block, set[int]] = field(default_factory=lambda: {"reasoning": set(), "text": set(), "call": set()})
    top: int = -1


class MessageConverter(BaseConverter):
    """Tracks model messages by source and tool calls by ``ToolCallDelta.index`` and id; the subclasses say how
    each step is written in their protocol."""

    def __init__(self, **options: Unpack[ConverterOptions]) -> None:
        super().__init__(**options)
        self._messages: dict[Hashable, Message] = {}
        #: Open tool calls by id, and the source whose message holds them.
        self._open_calls: dict[str, Hashable] = {}
        #: Every tool call id the browser has seen started.
        self._started: set[str] = set()
        #: Tool calls whose ids were made here, waiting for their results, oldest first.
        self._unanswered: list[Call] = []

    def convert(self, chunk: StreamingChunk, source: Hashable) -> list[Frame]:
        out: list[str] = []
        open_message = self._messages.get(source)
        if open_message is not None and _starts_another(open_message, chunk):
            # The model call that streamed the open message failed without a finish_reason, and the same caller
            # streams another one: FallbackChatGenerator retrying with the next generator. The two stay apart.
            del self._messages[source]
            out += self._close(open_message)
        reasoning = chunk.reasoning.reasoning_text if chunk.reasoning is not None else ""
        if reasoning:
            message = self._message(source, out)
            out += self._close_text(message)
            if message.reasoning_id is None:
                message.reasoning_id = new_id("rsn")
                out += self._start_reasoning(message.reasoning_id)
            out += self._reasoning_delta(message.reasoning_id, reasoning)
        if chunk.content:
            message = self._message(source, out)
            out += self._close_reasoning(message)
            if message.text_id is None:
                message.text_id = message.id if message.texts == 0 else new_id("msg")
                message.parent_id = message.text_id
                message.texts += 1
                out += self._start_text(message.text_id)
            out += self._text_delta(message.text_id, chunk.content)
        for delta in chunk.tool_calls or []:
            message = self._message(source, out)
            out += self._close_reasoning(message)
            out += self._close_text(message)
            call = message.calls.get(delta.index)
            if call is None:
                call = Call(delta.id or new_id("call"), delta.tool_name or "", generated=not delta.id)
                message.calls[delta.index] = call
                self._open_calls[call.id] = source
                self._started.add(call.id)
                out += self._start_call(call, message.parent_id)
            if delta.arguments:
                call.arguments.append(delta.arguments)
                out += self._call_delta(call, delta.arguments)
        if chunk.tool_call_result is not None:
            out += self._result(chunk.tool_call_result)
        message = self._messages.get(source)
        if message is not None and chunk.index is not None and (reasoning or chunk.content or chunk.tool_calls):
            message.blocks[_block(chunk)].add(chunk.index)
            message.blocks["call"].update(delta.index for delta in chunk.tool_calls or [])
            message.top = max(message.top, chunk.index)
        if chunk.finish_reason in MODEL_FINISH:
            message = self._messages.pop(source, None)
            if message is not None:
                out += self._close(message)
            out += self._message_finished(chunk.finish_reason)
        elif chunk.finish_reason == "tool_call_results":
            out += self._results_finished()
        return self._frames(out)

    # -- what the subclasses write ---------------------------------------------------------------------------------

    def _message_opened(self) -> list[str]:
        return []

    def _start_reasoning(self, reasoning_id: str) -> list[str]:
        raise NotImplementedError

    def _reasoning_delta(self, reasoning_id: str, delta: str) -> list[str]:
        raise NotImplementedError

    def _end_reasoning(self, reasoning_id: str) -> list[str]:
        raise NotImplementedError

    def _start_text(self, text_id: str) -> list[str]:
        raise NotImplementedError

    def _text_delta(self, text_id: str, delta: str) -> list[str]:
        raise NotImplementedError

    def _end_text(self, text_id: str) -> list[str]:
        raise NotImplementedError

    def _start_call(self, call: Call, parent_id: str) -> list[str]:
        raise NotImplementedError

    def _call_delta(self, call: Call, delta: str) -> list[str]:
        raise NotImplementedError

    def _end_calls(self, calls: list[Call]) -> list[str]:
        raise NotImplementedError

    def _replay_call(self, call: Call) -> list[str]:
        """A tool call the browser never saw started, written in full before its result."""
        raise NotImplementedError

    def _result_frames(self, call_id: str, result: ToolCallResult) -> list[str]:
        raise NotImplementedError

    def _message_finished(self, reason: str | None) -> list[str]:
        return []

    def _results_finished(self) -> list[str]:
        return []

    # -- bookkeeping -------------------------------------------------------------------------------------------------

    def _message(self, source: Hashable, out: list[str]) -> Message:
        message = self._messages.get(source)
        if message is None:
            message_id = new_id("msg")
            message = self._messages[source] = Message(message_id, message_id)
            out += self._message_opened()
        return message

    def _close_reasoning(self, message: Message) -> list[str]:
        if message.reasoning_id is None:
            return []
        reasoning_id, message.reasoning_id = message.reasoning_id, None
        return self._end_reasoning(reasoning_id)

    def _close_text(self, message: Message) -> list[str]:
        if message.text_id is None:
            return []
        text_id, message.text_id = message.text_id, None
        return self._end_text(text_id)

    def _close(self, message: Message) -> list[str]:
        out = self._close_reasoning(message) + self._close_text(message)
        calls = list(message.calls.values())
        message.calls = {}
        for call in calls:
            self._open_calls.pop(call.id, None)
            if call.generated:
                self._unanswered.append(call)
        return out + (self._end_calls(calls) if calls else [])

    def _close_all(self) -> list[str]:
        out: list[str] = []
        while self._messages:
            out += self._close(self._messages.pop(next(iter(self._messages))))
        return out

    def _result(self, result: ToolCallResult) -> list[str]:
        origin = result.origin
        out: list[str] = []
        call_id = origin.id
        source = self._open_calls.get(call_id) if call_id else None
        if source is not None:  # the generator never said its message was finished
            message = self._messages.pop(source, None)
            if message is not None:
                out += self._close(message)
        if not call_id:
            call_id = self._answered(origin)
        if call_id is None or call_id not in self._started:
            call = Call(call_id or new_id("call"), origin.tool_name, generated=call_id is None)
            call.arguments.append(json.dumps(origin.arguments, ensure_ascii=False))
            self._started.add(call.id)
            out += self._replay_call(call)
            call_id = call.id
        return out + self._result_frames(call_id, result)

    def _answered(self, origin: ToolCall) -> str | None:
        """The id of the waiting id-less call that ``origin`` answers: same tool and arguments, else same tool."""
        same_tool = [call for call in self._unanswered if call.name == origin.tool_name]
        for call in same_tool:
            try:
                if json.loads(call.text or "{}") == origin.arguments:
                    self._unanswered.remove(call)
                    return call.id
            except ValueError:
                continue
        if same_tool:
            self._unanswered.remove(same_tool[0])
            return same_tool[0].id
        return None


def _block(chunk: StreamingChunk) -> Block:
    if chunk.tool_calls:
        return "call"
    return "reasoning" if chunk.reasoning is not None and not chunk.content else "text"


def _starts_another(message: Message, chunk: StreamingChunk) -> bool:
    """Whether ``chunk``, from the caller whose ``message`` is open, begins another model message: a tool call delta
    at an index the message used, with another provider id, or without ids but named again after its arguments; or a
    block that starts at an index the message used for that kind (or for a tool call), or below its highest. OpenAI's
    chat completions number text and tool calls apart, both from 0, so a tool call does not count the text's index;
    some providers repeat a call's id and name on every fragment, so those alone mean nothing."""
    for delta in chunk.tool_calls or []:
        call = message.calls.get(delta.index)
        if call is None:
            continue
        if delta.id and not call.generated and delta.id != call.id:
            return True
        if not delta.id and call.generated and delta.tool_name and call.arguments:
            return True
    if not chunk.start or chunk.index is None or chunk.tool_call_result is not None:
        return False
    kind = _block(chunk)
    used = message.blocks[kind] if kind == "call" else message.blocks[kind] | message.blocks["call"]
    return chunk.index < message.top or chunk.index in used
