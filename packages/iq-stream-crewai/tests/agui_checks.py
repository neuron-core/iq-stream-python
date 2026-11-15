"""Checks of an AG-UI event sequence that need no Node: every event parses as an ``ag_ui.core`` model, and the
sequence follows the rules of ``@ag-ui/client``'s ``verifyEvents`` that the adapter can break."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import ag_ui.core as ag
from pydantic import TypeAdapter

_EVENT: TypeAdapter[Any] = TypeAdapter(ag.Event)


def parse(data: Iterable[str]) -> list[dict[str, Any]]:
    """Each frame's data as a dict, after checking that it is an AG-UI event."""
    events: list[dict[str, Any]] = []
    for text in data:
        _EVENT.validate_json(text)
        events.append(json.loads(text))
    return events


def types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def verify(events: list[dict[str, Any]]) -> None:
    """Raises AssertionError where ``verifyEvents`` would reject the sequence."""
    assert events, "no events"
    assert events[0]["type"] == "RUN_STARTED", events[0]
    texts: set[str] = set()
    reasoning: set[str] = set()
    spans: set[str] = set()
    tools: set[str] = set()
    started_tools: set[str] = set()
    steps: set[str] = set()
    finished = False
    for index, event in enumerate(events):
        kind = event["type"]
        assert not finished, f"event {index} ({kind}) after the run ended"
        if kind == "TEXT_MESSAGE_START":
            assert event["messageId"] not in texts, event
            texts.add(event["messageId"])
        elif kind in ("TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"):
            assert event["messageId"] in texts, event
            if kind == "TEXT_MESSAGE_END":
                texts.discard(event["messageId"])
        elif kind == "REASONING_START":
            assert event["messageId"] not in spans, event
            spans.add(event["messageId"])
        elif kind == "REASONING_END":
            assert event["messageId"] in spans, event
            spans.discard(event["messageId"])
        elif kind == "REASONING_MESSAGE_START":
            assert event["messageId"] not in reasoning, event
            reasoning.add(event["messageId"])
        elif kind in ("REASONING_MESSAGE_CONTENT", "REASONING_MESSAGE_END"):
            assert event["messageId"] in reasoning, event
            if kind == "REASONING_MESSAGE_END":
                reasoning.discard(event["messageId"])
        elif kind == "TOOL_CALL_START":
            assert event["toolCallId"] not in tools, event
            assert event["toolCallId"] not in started_tools, f"tool call id reused: {event}"
            tools.add(event["toolCallId"])
            started_tools.add(event["toolCallId"])
        elif kind in ("TOOL_CALL_ARGS", "TOOL_CALL_END"):
            assert event["toolCallId"] in tools, event
            if kind == "TOOL_CALL_END":
                tools.discard(event["toolCallId"])
        elif kind == "TOOL_CALL_RESULT":
            assert event["toolCallId"] in started_tools, event
            assert event["toolCallId"] not in tools, f"result before TOOL_CALL_END: {event}"
        elif kind == "STEP_STARTED":
            assert event["stepName"] not in steps, event
            steps.add(event["stepName"])
        elif kind == "STEP_FINISHED":
            assert event["stepName"] in steps, event
            steps.discard(event["stepName"])
        elif kind == "RUN_FINISHED":
            assert not (texts or reasoning or spans or tools or steps), (texts, reasoning, spans, tools, steps)
            finished = True
        elif kind == "RUN_ERROR":
            finished = True
        elif kind == "RUN_STARTED":
            assert index == 0, "a second RUN_STARTED in one run"
    assert finished, "no RUN_FINISHED or RUN_ERROR at the end"


@dataclass
class Conversation:
    """What a client reconstructs: text per message, tool calls with their arguments and results."""

    texts: dict[str, str] = field(default_factory=lambda: {})
    names: dict[str, str] = field(default_factory=lambda: {})
    tool_names: dict[str, str] = field(default_factory=lambda: {})
    tool_args: dict[str, str] = field(default_factory=lambda: {})
    tool_results: dict[str, str] = field(default_factory=lambda: {})
    reasoning: dict[str, str] = field(default_factory=lambda: {})

    @property
    def text(self) -> str:
        return "".join(self.texts.values())


def conversation(events: list[dict[str, Any]]) -> Conversation:
    result = Conversation()
    for event in events:
        kind = event["type"]
        if kind == "TEXT_MESSAGE_START":
            result.texts.setdefault(event["messageId"], "")
            if "name" in event:
                result.names[event["messageId"]] = event["name"]
        elif kind == "TEXT_MESSAGE_CONTENT":
            result.texts[event["messageId"]] += event["delta"]
        elif kind == "REASONING_MESSAGE_START":
            result.reasoning.setdefault(event["messageId"], "")
        elif kind == "REASONING_MESSAGE_CONTENT":
            result.reasoning[event["messageId"]] += event["delta"]
        elif kind == "TOOL_CALL_START":
            result.tool_names[event["toolCallId"]] = event["toolCallName"]
            result.tool_args[event["toolCallId"]] = ""
        elif kind == "TOOL_CALL_ARGS":
            result.tool_args[event["toolCallId"]] += event["delta"]
        elif kind == "TOOL_CALL_RESULT":
            result.tool_results[event["toolCallId"]] = event["content"]
    return result


def checked(data: Iterable[str]) -> list[dict[str, Any]]:
    """Parsed and verified."""
    events = parse(data)
    verify(events)
    return events
