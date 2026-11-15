"""What a run publishes: one frame per stream part, in the order the graph yields them, and what relay() yields."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fakes import ASYNC_TOKENS, FINAL, FIRST, TOOL_RESULT, USER, text_of, weather_agent
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher

from iq_stream_langchain import DEFAULT_STREAM_MODE, apublish_run, arelay, publish_run, relay

EXPECTED_TYPES = ["metadata", *["messages"] * 9, "updates", "tools", "tools", "messages", "updates"]
EXPECTED_TYPES += [*["messages"] * 8, "updates"]
# An async run streams whole messages on Python 3.10 (see fakes.ASYNC_TOKENS).
EXPECTED_ASYNC_TYPES = EXPECTED_TYPES if ASYNC_TOKENS else ["metadata", "messages", "updates", "tools", "tools"]
EXPECTED_ASYNC_TYPES += [] if ASYNC_TOKENS else ["messages", "updates", "messages", "updates"]


def decoded(publisher: RecordingPublisher | AsyncRecordingPublisher) -> list[dict[str, Any]]:
    assert all(frame.event is None for frame in publisher.frames), "frames carry no event name"
    return [json.loads(data) for data in publisher.data]


def test_a_sync_run_publishes_every_part_in_order_and_ends() -> None:
    publisher = RecordingPublisher()
    with publisher:
        yielded = list(relay(publisher, weather_agent(), USER))

    parts = decoded(publisher)
    assert [part["type"] for part in parts] == EXPECTED_TYPES
    assert text_of(parts) == FIRST + FINAL
    tool_messages = [part for part in parts if part["type"] == "messages" and part["data"][0]["type"] == "tool"]
    assert [message["data"][0]["content"] for message in tool_messages] == [TOOL_RESULT]
    tools = [part["data"] for part in parts if part["type"] == "tools"]
    assert [(tool["event"], tool["tool_call_id"]) for tool in tools] == [
        ("tool-started", "call_1"),
        ("tool-finished", "call_1"),
    ]
    assert tools[0]["input"] == {"city": "Turin"}
    assert all(part["ns"] == [] for part in parts)
    assert publisher.outcome == "end"
    # Every part was yielded, and each published one is the same part.
    assert [part["type"] for part in yielded] == EXPECTED_TYPES[1:]


async def test_an_async_run_publishes_the_same_sequence() -> None:
    publisher = AsyncRecordingPublisher()
    async with publisher:
        yielded = [part async for part in arelay(publisher, weather_agent(), USER)]

    parts = decoded(publisher)
    assert [part["type"] for part in parts] == EXPECTED_ASYNC_TYPES
    assert text_of(parts) == FIRST + FINAL
    assert len(yielded) == len(parts) - 1
    assert publisher.outcome == "end"


def test_the_first_frame_names_the_run_and_the_thread() -> None:
    publisher = RecordingPublisher()
    config: Any = {"configurable": {"thread_id": "thread-7"}, "run_id": "0b9e6a52-6c1e-4d4f-9a59-0c5c3f5a9e11"}
    with publisher:
        list(relay(publisher, weather_agent(), USER, config))
    first = json.loads(publisher.data[0])
    assert first == {
        "type": "metadata",
        "ns": [],
        "data": {"run_id": "0b9e6a52-6c1e-4d4f-9a59-0c5c3f5a9e11", "thread_id": "thread-7"},
    }

    publisher = RecordingPublisher()
    with publisher:
        list(relay(publisher, weather_agent(), USER))
    assert json.loads(publisher.data[0])["data"] == {"run_id": None, "thread_id": None}


def test_relay_yields_exactly_what_the_graph_yields() -> None:
    expected = list(weather_agent().stream(USER, stream_mode=list(DEFAULT_STREAM_MODE), subgraphs=True, version="v2"))
    publisher = RecordingPublisher()
    with publisher:
        yielded = list(relay(publisher, weather_agent(), USER))
    assert [(part["type"], part["ns"]) for part in yielded] == [(part["type"], part["ns"]) for part in expected]


def test_also_yield_modes_are_yielded_and_not_published() -> None:
    publisher = RecordingPublisher()
    with publisher:
        yielded = list(relay(publisher, weather_agent(), USER, also_yield=("values",)))
    assert sum(part["type"] == "values" for part in yielded) == 4
    assert "values" not in [part["type"] for part in decoded(publisher)]
    assert len(publisher.frames) == len(EXPECTED_TYPES)


def test_stream_mode_chooses_what_is_published() -> None:
    publisher = RecordingPublisher()
    with publisher:
        yielded = list(relay(publisher, weather_agent(), USER, stream_mode=("updates", "values")))
    assert {part["type"] for part in yielded} == {"updates", "values"}
    assert [part["type"] for part in decoded(publisher)][:3] == ["metadata", "values", "updates"]


def test_publish_run_returns_the_final_state_and_publishes_no_values() -> None:
    publisher = RecordingPublisher()
    with publisher:
        final = publish_run(publisher, weather_agent(), USER)
        assert publisher.outcome is None, "the adapter leaves the close to the block"
    assert final["messages"][-1].content == FINAL
    assert [part["type"] for part in decoded(publisher)] == EXPECTED_TYPES
    assert publisher.outcome == "end"


async def test_apublish_run_returns_the_final_state() -> None:
    publisher = AsyncRecordingPublisher()
    async with publisher:
        final = await apublish_run(publisher, weather_agent(), USER)
    assert final["messages"][-1].content == FINAL
    assert [part["type"] for part in decoded(publisher)] == EXPECTED_ASYNC_TYPES
    assert publisher.outcome == "end"


async def test_arelay_takes_a_synchronous_publisher_too() -> None:
    publisher = RecordingPublisher()
    with publisher:
        final = await apublish_run(publisher, weather_agent(), USER)
    assert final["messages"][-1].content == FINAL
    assert len(publisher.frames) == len(EXPECTED_ASYNC_TYPES)


def test_stream_options_go_to_the_graph() -> None:
    publisher = RecordingPublisher()
    with publisher:
        parts = list(relay(publisher, weather_agent(), USER, interrupt_before=["tools"]))
    assert parts[-1]["type"] == "updates" and "__interrupt__" in parts[-1]["data"]
    assert "tools" not in [part["type"] for part in parts]


def test_mistakes_are_refused_at_the_call() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent()
    with pytest.raises(ValueError, match="close_on_cancel"):
        relay(publisher, agent, USER, close_on_cancel="later")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata"):
        relay(publisher, agent, USER, metadata="some")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequences"):
        relay(publisher, agent, USER, stream_mode="messages")
    with pytest.raises(TypeError, match="arelay"):
        relay(AsyncRecordingPublisher(), agent, USER)  # type: ignore[arg-type]
    assert publisher.frames == []
