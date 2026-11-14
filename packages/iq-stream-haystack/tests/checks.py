"""Checks of what a browser would receive, in Python: every AG-UI event parses with the ``ag_ui.core`` models and the
sequence follows the rules the TypeScript client's ``verifyEvents`` enforces; AI SDK chunks follow the order
``useChat`` needs. The same frames go through the real JavaScript libraries in the integration tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ag_ui.core import BaseEvent, Event
from pydantic import TypeAdapter

_EVENT: TypeAdapter[BaseEvent] = TypeAdapter(Event)  # pyright: ignore[reportAssignmentType]


def parse_agui(data: list[str]) -> list[BaseEvent]:
    return [_EVENT.validate_json(item) for item in data]


@dataclass
class Conversation:
    """What a reducer rebuilds from a run's events."""

    texts: list[str] = field(default_factory=lambda: [])
    reasoning: list[str] = field(default_factory=lambda: [])
    #: (id, name, arguments as text, parent message id)
    calls: list[tuple[str, str, str, str | None]] = field(default_factory=lambda: [])
    #: tool call id -> content
    results: dict[str, str] = field(default_factory=lambda: {})
    custom: list[tuple[str, Any]] = field(default_factory=lambda: [])
    terminal: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def text(self) -> str:
        return "".join(self.texts)


def verify_agui(data: list[str]) -> Conversation:
    """Parses and checks the events; returns the conversation they describe."""
    events = [json.loads(item) for item in data]
    parse_agui(data)
    assert events and events[0]["type"] == "RUN_STARTED", events[:1]
    conversation = Conversation()
    texts: dict[str, list[str]] = {}
    reasoning: dict[str, list[str]] = {}
    open_texts: set[str] = set()
    open_reasoning: set[str] = set()
    calls: dict[str, list[str]] = {}
    open_calls: set[str] = set()
    ended_calls: set[str] = set()
    for number, event in enumerate(events):
        kind = event["type"]
        assert not conversation.terminal, f"{kind} after the terminal event"
        if kind == "RUN_STARTED":
            assert number == 0, "RUN_STARTED twice"
        elif kind == "TEXT_MESSAGE_START":
            assert event["messageId"] not in texts, "a text message started twice"
            texts[event["messageId"]] = []
            open_texts.add(event["messageId"])
        elif kind == "TEXT_MESSAGE_CONTENT":
            assert event["messageId"] in open_texts, "text content outside its message"
            assert event["delta"], "an empty delta"
            texts[event["messageId"]].append(event["delta"])
        elif kind == "TEXT_MESSAGE_END":
            assert event["messageId"] in open_texts, "a text message ended twice or never started"
            open_texts.discard(event["messageId"])
            conversation.texts.append("".join(texts[event["messageId"]]))
        elif kind == "REASONING_START":
            assert event["messageId"] not in reasoning
            reasoning[event["messageId"]] = []
            open_reasoning.add(event["messageId"])
        elif kind == "REASONING_MESSAGE_START":
            assert event["messageId"] in open_reasoning
        elif kind == "REASONING_MESSAGE_CONTENT":
            assert event["messageId"] in open_reasoning
            reasoning[event["messageId"]].append(event["delta"])
        elif kind == "REASONING_MESSAGE_END":
            assert event["messageId"] in open_reasoning
        elif kind == "REASONING_END":
            assert event["messageId"] in open_reasoning
            open_reasoning.discard(event["messageId"])
            conversation.reasoning.append("".join(reasoning[event["messageId"]]))
        elif kind == "TOOL_CALL_START":
            assert event["toolCallId"] not in calls, "a tool call started twice"
            calls[event["toolCallId"]] = []
            open_calls.add(event["toolCallId"])
            conversation.calls.append((event["toolCallId"], event["toolCallName"], "", event.get("parentMessageId")))
        elif kind == "TOOL_CALL_ARGS":
            assert event["toolCallId"] in open_calls, "arguments outside their tool call"
            calls[event["toolCallId"]].append(event["delta"])
        elif kind == "TOOL_CALL_END":
            assert event["toolCallId"] in open_calls, "a tool call ended twice or never started"
            open_calls.discard(event["toolCallId"])
            ended_calls.add(event["toolCallId"])
        elif kind == "TOOL_CALL_RESULT":
            assert event["toolCallId"] in ended_calls, "a result before its tool call ended"
            assert event["toolCallId"] not in conversation.results, "two results for one tool call"
            conversation.results[event["toolCallId"]] = event["content"]
        elif kind == "CUSTOM":
            conversation.custom.append((event["name"], event["value"]))
        elif kind in ("RUN_FINISHED", "RUN_ERROR"):
            assert not (open_texts or open_reasoning or open_calls), "the run ended with open messages"
            conversation.terminal = event
        else:
            raise AssertionError(f"unexpected event {kind}")
    conversation.calls = [(cid, name, "".join(calls[cid]), parent) for cid, name, _, parent in conversation.calls]
    return conversation


@dataclass
class UIMessage:
    """What ``useChat`` rebuilds from a run's UI message chunks."""

    texts: list[str] = field(default_factory=lambda: [])
    reasoning: list[str] = field(default_factory=lambda: [])
    #: id -> (name, input, output or error text, state)
    tools: dict[str, dict[str, Any]] = field(default_factory=lambda: {})
    steps: int = 0
    data: list[tuple[str, Any]] = field(default_factory=lambda: [])
    terminal: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def text(self) -> str:
        return "".join(self.texts)


def verify_ai_sdk(data: list[str]) -> UIMessage:
    chunks = [json.loads(item) for item in data]
    assert chunks and chunks[0]["type"] == "start", chunks[:1]
    message = UIMessage()
    texts: dict[str, list[str]] = {}
    reasoning: dict[str, list[str]] = {}
    step = False
    for chunk in chunks[1:]:
        kind = chunk["type"]
        assert not message.terminal, f"{kind} after the terminal chunk"
        if kind == "start-step":
            assert not step, "start-step inside a step"
            step = True
            message.steps += 1
        elif kind == "finish-step":
            assert step, "finish-step outside a step"
            assert not texts and not reasoning, "finish-step with open text or reasoning"
            step = False
        elif kind == "text-start":
            assert chunk["id"] not in texts
            texts[chunk["id"]] = []
        elif kind == "text-delta":
            texts[chunk["id"]].append(chunk["delta"])
        elif kind == "text-end":
            message.texts.append("".join(texts.pop(chunk["id"])))
        elif kind == "reasoning-start":
            reasoning[chunk["id"]] = []
        elif kind == "reasoning-delta":
            reasoning[chunk["id"]].append(chunk["delta"])
        elif kind == "reasoning-end":
            message.reasoning.append("".join(reasoning.pop(chunk["id"])))
        elif kind == "tool-input-start":
            assert chunk["toolCallId"] not in message.tools
            message.tools[chunk["toolCallId"]] = {"name": chunk["toolName"], "deltas": [], "state": "input-streaming"}
        elif kind == "tool-input-delta":
            message.tools[chunk["toolCallId"]]["deltas"].append(chunk["inputTextDelta"])
        elif kind == "tool-input-available":
            tool = message.tools[chunk["toolCallId"]]
            tool.update(input=chunk["input"], state="input-available")
        elif kind == "tool-output-available":
            message.tools[chunk["toolCallId"]].update(output=chunk["output"], state="output-available")
        elif kind == "tool-output-error":
            message.tools[chunk["toolCallId"]].update(error=chunk["errorText"], state="output-error")
        elif kind.startswith("data-"):
            message.data.append((kind[5:], chunk["data"]))
        elif kind in ("finish", "error", "abort"):
            assert not texts and not reasoning, "the run ended with open parts"
            assert not step, "the run ended inside a step"
            message.terminal = chunk
        else:
            raise AssertionError(f"unexpected chunk {kind}")
    return message
