"""Which close line a run gets: end when the answer finished or waits for a human, fail when it failed or was
cancelled (end on request), and the frame before it that tells the browser why."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from fakes import USER, FakeToolCallingModel, failing_agent, tool_call, weather_agent
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command, interrupt

from iq_stream_langchain import CANCELLED_PART, MASKED_MESSAGE, arelay, publish_run, relay


def last(publisher: RecordingPublisher | AsyncRecordingPublisher) -> dict[str, Any]:
    return json.loads(publisher.data[-1])


def test_a_failing_run_publishes_its_error_and_closes_fail() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError, match="exploded"), publisher:
        for _ in relay(publisher, failing_agent(), USER):
            pass
    assert last(publisher) == {"type": "error", "ns": [], "data": {"error": "RuntimeError", "message": MASKED_MESSAGE}}
    assert [json.loads(data)["type"] for data in publisher.data] == ["metadata", "messages", "messages", "messages", "error"]
    assert publisher.outcome == "fail"


def test_the_error_message_is_published_only_when_asked() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError), publisher:
        publish_run(publisher, failing_agent(), USER, expose_errors=True)
    assert last(publisher)["data"] == {"error": "RuntimeError", "message": "fake model exploded mid-stream"}
    assert publisher.outcome == "fail"


async def test_an_async_failing_run_closes_fail() -> None:
    publisher = AsyncRecordingPublisher()
    with pytest.raises(RuntimeError):
        async with publisher:
            async for _ in arelay(publisher, failing_agent(), USER):
                pass
    assert last(publisher)["data"]["error"] == "RuntimeError"
    assert publisher.outcome == "fail"


async def run_and_cancel(publisher: AsyncRecordingPublisher, **options: Any) -> None:
    started = asyncio.Event()

    async def run() -> None:
        async with publisher:
            async for part in arelay(publisher, weather_agent(token_delay=0.05), USER, **options):
                if part["type"] == "messages":
                    started.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cancelled_task_publishes_the_cancel_and_closes_fail() -> None:
    publisher = AsyncRecordingPublisher()
    await run_and_cancel(publisher)
    assert publisher.data[-1] == CANCELLED_PART
    assert last(publisher) == {"type": "error", "ns": [], "data": {"error": "Cancelled", "message": "run cancelled"}}
    assert publisher.outcome == "fail"
    assert publisher.data.count(CANCELLED_PART) == 1


async def test_a_cancelled_task_closes_end_when_asked() -> None:
    publisher = AsyncRecordingPublisher()
    await run_and_cancel(publisher, cancelled="end")
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "end"


async def test_a_cancelled_apublish_run_closes_fail() -> None:
    publisher = AsyncRecordingPublisher()

    async def run() -> None:
        async with publisher:
            from iq_stream_langchain import apublish_run

            await apublish_run(publisher, weather_agent(token_delay=0.05), USER)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "fail"


def test_a_sync_loop_left_early_is_a_cancelled_run() -> None:
    publisher = RecordingPublisher()
    with publisher:
        for index, _ in enumerate(relay(publisher, weather_agent(), USER)):
            if index == 2:
                break
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "fail"
    assert publisher.close_calls == 2, "the relay closed, and the block's own close changed nothing"

    publisher = RecordingPublisher()
    with publisher:
        for index, _ in enumerate(relay(publisher, weather_agent(), USER, cancelled="end")):
            if index == 2:
                break
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "end"


async def test_an_async_loop_left_early_under_aclosing_is_a_cancelled_run() -> None:
    publisher = AsyncRecordingPublisher()
    async with publisher:
        async with contextlib.aclosing(arelay(publisher, weather_agent(), USER)) as parts:
            index = 0
            async for _ in parts:
                index += 1
                if index == 3:
                    break
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "fail"


async def test_an_async_loop_left_early_without_aclosing_closes_end() -> None:
    """The limit the README states: Python finalizes the generator after the block has closed."""
    publisher = AsyncRecordingPublisher()
    parts = arelay(publisher, weather_agent(), USER)
    async with publisher:
        index = 0
        async for _ in parts:
            index += 1
            if index == 3:
                break
    assert publisher.outcome == "end"
    await parts.aclose()
    assert publisher.outcome == "end"


def test_a_drain_request_is_a_cancelled_run() -> None:
    control = RunControl()

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        control.request_drain("deploy")
        return f"sunny, 21C in {city}"

    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Sunny."]  # type: ignore[list-item]
        )
    )
    agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())
    publisher = RecordingPublisher()
    with pytest.raises(GraphDrained), publisher:
        publish_run(publisher, agent, USER, {"configurable": {"thread_id": "drain"}}, control=control)
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "fail"


@tool
def book(city: str) -> str:
    """Book a trip; a human approves it first."""
    approved = interrupt({"question": f"book {city}?"})
    return f"booked {city}: {approved}"


def booking_agent() -> Any:
    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("book", {"city": "Turin"}, "c1")]), "Booked."]  # type: ignore[list-item]
        )
    )
    return create_agent(model, [book], checkpointer=InMemorySaver())


def test_a_run_that_waits_for_a_human_closes_end_and_resumes_on_a_new_stream() -> None:
    agent = booking_agent()
    config: Any = {"configurable": {"thread_id": "hitl"}}

    paused = RecordingPublisher()
    with paused:
        publish_run(paused, agent, USER, config)
    parts = [json.loads(data) for data in paused.data]
    assert parts[-1]["type"] == "updates"
    assert parts[-1]["data"]["__interrupt__"][0]["value"] == {"question": "book Turin?"}
    assert parts[-1]["data"]["__interrupt__"][0]["id"]
    assert [part["data"]["event"] for part in parts if part["type"] == "tools"] == ["tool-started"]
    assert paused.outcome == "end"

    resumed = RecordingPublisher()
    with resumed:
        final = publish_run(resumed, agent, Command(resume="yes"), config)
    parts = [json.loads(data) for data in resumed.data]
    assert [part["data"]["event"] for part in parts if part["type"] == "tools"] == ["tool-started", "tool-finished"]
    assert final["messages"][-2].content == "booked Turin: yes"
    assert resumed.outcome == "end"
