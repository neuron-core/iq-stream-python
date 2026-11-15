"""Concurrent nodes, sub-agents and tool calls: one sequence on the wire, the iterator's own, each run's frames
told apart by namespace and tool call id."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import pytest
from fakes import (
    ASYNC_TOKENS,
    LANGGRAPH_1_2,
    USER,
    FakeToolCallingModel,
    get_weather,
    text_of,
    tool_call,
    two_sub_agents,
    weather_agent,
)
from iq_stream import Outcome, PublishResult
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from iq_stream_langchain import apublish_run, arelay, relay


def assert_interleaved(parts: list[dict[str, Any]]) -> None:
    order = "".join(
        part["ns"][0][0] for part in parts if part["type"] == "messages" and part["data"][0]["type"] == "AIMessageChunk"
    )
    assert "ab" in order and "ba" in order, f"the sub-agents' tokens interleave: {order}"
    assert text_of(parts, "a:") == "alpha says one two three four five"
    assert text_of(parts, "b:") == "beta says one two three four five"


def test_two_sub_agents_in_threads_interleave_and_each_text_is_intact() -> None:
    publisher = RecordingPublisher()
    with publisher:
        yielded = [
            (part["type"], list(part["ns"]))
            for part in relay(publisher, two_sub_agents(async_nodes=False), {"messages": []})
        ]
    parts = [json.loads(data) for data in publisher.data][1:]
    assert [(part["type"], part["ns"]) for part in parts] == yielded, "published in the order yielded"
    assert_interleaved(parts)
    assert publisher.outcome == "end"


@pytest.mark.skipif(not ASYNC_TOKENS, reason="LangChain streams tokens from async nodes on Python 3.11+ only")
async def test_two_async_sub_agents_interleave_and_each_text_is_intact() -> None:
    publisher = AsyncRecordingPublisher()
    yielded: list[tuple[str, list[str]]] = []
    async with publisher:
        async for part in arelay(publisher, two_sub_agents(async_nodes=True), {"messages": []}):
            yielded.append((part["type"], list(part["ns"])))
    parts = [json.loads(data) for data in publisher.data][1:]
    assert [(part["type"], part["ns"]) for part in parts] == yielded, "published in the order yielded"
    assert_interleaved(parts)
    assert publisher.outcome == "end"


def parallel_agent() -> Any:
    model = FakeToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[  # type: ignore[list-item]
                        tool_call("get_weather", {"city": "Turin"}, "c1"),
                        tool_call("get_weather", {"city": "Rome"}, "c2"),
                    ],
                ),
                AIMessage(content="Both sunny."),
            ]
        )
    )
    return create_agent(model, [get_weather])


def tool_events(publisher: RecordingPublisher | AsyncRecordingPublisher) -> list[tuple[str, str]]:
    parts = [json.loads(data) for data in publisher.data]
    return [(part["data"]["event"], part["data"]["tool_call_id"]) for part in parts if part["type"] == "tools"]


def assert_each_call_ordered(events: list[tuple[str, str]]) -> None:
    assert sorted(events) == sorted(
        [("tool-started", "c1"), ("tool-finished", "c1"), ("tool-started", "c2"), ("tool-finished", "c2")]
    )
    for call in ("c1", "c2"):
        assert events.index(("tool-started", call)) < events.index(("tool-finished", call))


def test_parallel_tool_calls_in_a_sync_run() -> None:
    publisher = RecordingPublisher()
    with publisher:
        list(relay(publisher, parallel_agent(), USER))
    if LANGGRAPH_1_2:  # the tools stream mode
        assert_each_call_ordered(tool_events(publisher))
    parts = [json.loads(data) for data in publisher.data]
    results = [
        part["data"][0]["content"] for part in parts if part["type"] == "messages" and part["data"][0]["type"] == "tool"
    ]
    assert sorted(results) == ["sunny, 21C in Rome", "sunny, 21C in Turin"]


async def test_parallel_tool_calls_in_an_async_run() -> None:
    publisher = AsyncRecordingPublisher()
    async with publisher:
        await apublish_run(publisher, parallel_agent(), USER)
    if LANGGRAPH_1_2:
        assert_each_call_ordered(tool_events(publisher))
    parts = [json.loads(data) for data in publisher.data]
    results = [
        part["data"][0]["content"] for part in parts if part["type"] == "messages" and part["data"][0]["type"] == "tool"
    ]
    assert sorted(results) == ["sunny, 21C in Rome", "sunny, 21C in Turin"]


async def test_two_runs_at_once_keep_their_own_frames() -> None:
    first, second = AsyncRecordingPublisher(), AsyncRecordingPublisher()

    async def run(publisher: AsyncRecordingPublisher, agent: Any) -> None:
        async with publisher:
            await apublish_run(publisher, agent, USER)

    alpha = create_agent(FakeToolCallingModel(messages=iter(["alpha one two three"]), token_delay=0.01), [])
    await asyncio.gather(run(first, alpha), run(second, weather_agent(token_delay=0.01)))
    assert text_of([json.loads(data) for data in first.data]) == "alpha one two three"
    assert text_of([json.loads(data) for data in second.data]).endswith("21 degrees.")
    assert first.outcome == second.outcome == "end"


class SlowClosePublisher(RecordingPublisher):
    """A synchronous publisher whose close waits for the server, as the real one does."""

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        time.sleep(1.0)
        return super().close(outcome, timeout)


async def test_a_synchronous_publisher_is_closed_off_the_event_loop() -> None:
    """arelay() takes a synchronous Publisher; closing it on a cancel must not stall the application's other tasks."""
    publisher = SlowClosePublisher()
    started = asyncio.Event()
    gaps: list[float] = []

    async def tick() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    async def run() -> None:
        async for part in arelay(publisher, weather_agent(token_delay=0.05), USER):
            if part["type"] == "messages":
                started.set()

    ticker = asyncio.create_task(tick())
    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 5)
    gaps.clear()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    ticker.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticker
    assert publisher.outcome == "fail"
    assert sum(gaps) > 0.9, "the close took its second"
    assert max(gaps) < 0.5, "and the loop went on meanwhile"
