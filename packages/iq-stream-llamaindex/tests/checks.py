"""Assertions on what was published, shared by both suites: AG-UI events validated with the ``ag_ui.core`` models
and checked for the order a client needs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ag_ui.core import Event as AGUIEvent
from pydantic import TypeAdapter

_EVENT = TypeAdapter[Any](AGUIEvent)


def agui_events(data: Iterable[str]) -> list[dict[str, Any]]:
    """Each frame's data validated as one AG-UI event, and parsed."""
    events: list[dict[str, Any]] = []
    for text in data:
        _EVENT.validate_json(text)
        events.append(json.loads(text))
    return events


def types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def assert_well_formed(events: list[dict[str, Any]]) -> None:
    """RUN_STARTED first and one terminal event last; each message opened, filled and closed in order; each tool call's
    start, arguments and end contiguous, and one result after its end; steps balanced, none open at RUN_FINISHED."""
    kinds = types(events)
    assert kinds[0] == "RUN_STARTED"
    assert kinds[-1] in ("RUN_FINISHED", "RUN_ERROR")
    assert sum(kind in ("RUN_FINISHED", "RUN_ERROR") for kind in kinds) == 1
    opened: dict[str, str] = {}  # message id -> the kind that opened it
    closed: set[str] = set()
    step: str | None = None
    tools_ended: set[str] = set()
    results: dict[str, int] = {}
    for index, event in enumerate(events):
        kind = event["type"]
        if kind in ("TEXT_MESSAGE_START", "REASONING_MESSAGE_START"):
            assert event["messageId"] not in opened
            opened[event["messageId"]] = kind
        elif kind in ("TEXT_MESSAGE_CONTENT", "REASONING_MESSAGE_CONTENT"):
            assert event["messageId"] in opened and event["messageId"] not in closed
        elif kind in ("TEXT_MESSAGE_END", "REASONING_MESSAGE_END"):
            assert event["messageId"] in opened and event["messageId"] not in closed
            closed.add(event["messageId"])
        elif kind == "TOOL_CALL_START":
            following = kinds[index + 1 : index + 3]
            assert following[0] == "TOOL_CALL_ARGS" and following[-1] in ("TOOL_CALL_ARGS", "TOOL_CALL_END")
            call = event["toolCallId"]
            rest = events[index + 1 :]
            end = next(i for i, later in enumerate(rest) if later["type"] == "TOOL_CALL_END")
            assert all(later["type"] == "TOOL_CALL_ARGS" and later["toolCallId"] == call for later in rest[:end])
            assert rest[end]["toolCallId"] == call
            tools_ended.add(call)
        elif kind == "TOOL_CALL_RESULT":
            assert event["toolCallId"] in tools_ended
            results[event["toolCallId"]] = results.get(event["toolCallId"], 0) + 1
        elif kind == "STEP_STARTED":
            assert step is None
            step = event["stepName"]
        elif kind == "STEP_FINISHED":
            assert step == event["stepName"]
            step = None
        elif kind == "RUN_FINISHED":
            assert step is None
            assert set(opened) == closed
    assert all(count == 1 for count in results.values())


def joined(events: list[dict[str, Any]], kind: str) -> str:
    """The deltas of every event of one kind, concatenated."""
    return "".join(event["delta"] for event in events if event["type"] == kind)
