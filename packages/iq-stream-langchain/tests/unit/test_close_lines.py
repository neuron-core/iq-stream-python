"""Which close line a run gets: end when the answer finished or waits for a human, fail when it failed or was
cancelled (end on request), and the frame before it that tells the browser why."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from typing import Any

import pytest
from fakes import USER, FakeToolCallingModel, booking_agent, failing_agent, stubborn_agent, tool_call, weather_agent
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command

from iq_stream_langchain import CANCELLED_PART, MASKED_MESSAGE, apublish_run, arelay, publish_run, relay

CANCELLING = sys.version_info >= (3, 11)


def last(publisher: RecordingPublisher | AsyncRecordingPublisher) -> dict[str, Any]:
    return json.loads(publisher.data[-1])


def test_a_failing_run_publishes_its_error_and_closes_fail() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError, match="exploded"), publisher:
        for _ in relay(publisher, failing_agent(), USER):
            pass
    assert last(publisher) == {"type": "error", "ns": [], "data": {"error": "RuntimeError", "message": MASKED_MESSAGE}}
    assert [json.loads(data)["type"] for data in publisher.data] == [
        "metadata",
        "messages",
        "messages",
        "messages",
        "error",
    ]
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
    await run_and_cancel(publisher, close_on_cancel="end")
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "end"


async def test_a_cancelled_apublish_run_closes_fail() -> None:
    publisher = AsyncRecordingPublisher()

    async def run() -> None:
        async with publisher:
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
        for index, _ in enumerate(relay(publisher, weather_agent(), USER, close_on_cancel="end")):
            if index == 2:
                break
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "end"


async def test_an_async_loop_left_early_under_aclosing_is_a_cancelled_run() -> None:
    publisher = AsyncRecordingPublisher()
    async with publisher, contextlib.aclosing(arelay(publisher, weather_agent(), USER)) as parts:
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


def test_a_run_that_waits_for_a_human_closes_end_and_resumes_on_a_new_stream() -> None:
    agent = booking_agent(checkpointer=InMemorySaver())
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


def test_publish_run_returns_the_interrupt_as_invoke_does() -> None:
    final = publish_run(
        RecordingPublisher(), booking_agent(checkpointer=InMemorySaver()), USER, {"configurable": {"thread_id": "a"}}
    )
    invoked = booking_agent(checkpointer=InMemorySaver()).invoke(USER, {"configurable": {"thread_id": "b"}})
    assert set(final) == set(invoked) == {"messages", "__interrupt__"}
    assert isinstance(final["__interrupt__"], list)
    assert [item.value for item in final["__interrupt__"]] == [item.value for item in invoked["__interrupt__"]]
    assert final["__interrupt__"][0].value == {"question": "book Turin?"}

    finished = publish_run(RecordingPublisher(), weather_agent(), USER)
    assert "__interrupt__" not in finished


class SaveFailed(Exception):
    pass


@pytest.mark.parametrize("cancelled", ["fail", "end"])
def test_an_exception_of_the_callers_loop_never_closes_end(cancelled: Any) -> None:
    """A generator cannot tell it from a break, so the page reads the Cancelled part; the close line stays fail."""
    publisher = RecordingPublisher()
    with pytest.raises(SaveFailed), publisher:
        for part in relay(publisher, weather_agent(), USER, close_on_cancel=cancelled):
            if part["type"] == "updates":
                raise SaveFailed("the database is down")
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "fail"


@pytest.mark.parametrize("cancelled", ["fail", "end"])
async def test_an_exception_of_the_callers_async_loop_never_closes_end(cancelled: Any) -> None:
    publisher = AsyncRecordingPublisher()
    with pytest.raises(SaveFailed):
        async with (
            publisher,
            contextlib.aclosing(arelay(publisher, weather_agent(), USER, close_on_cancel=cancelled)) as parts,
        ):
            async for part in parts:
                if part["type"] == "updates":
                    raise SaveFailed("the database is down")
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "fail"


class TimedPublisher(RecordingPublisher):
    """Notes when the Cancelled part went out."""

    cancelled_at: float | None = None

    def send(self, data: str, event: str | None = None) -> bool:
        if data == CANCELLED_PART:
            self.cancelled_at = time.monotonic()
        return super().send(data, event)


def test_a_sync_loop_left_during_a_slow_tool_is_cancelled_at_once() -> None:
    """Stopping the graph waits for the node that runs in its thread; the page hears of the cancel before that."""

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        time.sleep(1.5)
        return "sunny"

    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Sunny."]  # type: ignore[list-item]
        )
    )
    publisher = TimedPublisher()
    left = 0.0
    with publisher:
        for part in relay(publisher, create_agent(model, [get_weather]), USER):
            if part["type"] == "tools":
                left = time.monotonic()
                break
        assert publisher.outcome == "fail", "closed before the block's own close"
        stopped = time.monotonic()
    assert publisher.cancelled_at is not None
    assert publisher.cancelled_at - left < 0.5
    assert stopped - left > 1.0, "the graph was stopped after its node ended"


@pytest.mark.skipif(not CANCELLING, reason="Python 3.10 has no Task.cancelling(): a cancel in the loop body is a break")
async def test_a_cancel_that_lands_in_the_loop_body_closes_with_close_on_cancel() -> None:
    """A stop route that cancels a task whose loop awaits something of its own (a websocket send, a save per part)."""
    publisher = AsyncRecordingPublisher()
    streaming = asyncio.Event()

    async def run() -> None:
        relayed = arelay(publisher, weather_agent(token_delay=0.02), USER, close_on_cancel="end")
        async with publisher, contextlib.aclosing(relayed):
            async for part in relayed:
                if part["type"] == "messages":
                    streaming.set()
                await asyncio.sleep(0.05)

    task = asyncio.create_task(run())
    await asyncio.wait_for(streaming.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert publisher.data[-1] == CANCELLED_PART
    assert publisher.outcome == "end"


class AsyncTimedPublisher(AsyncRecordingPublisher):
    """Notes when the Cancelled part went out and when the stream was first closed."""

    cancelled_at: float | None = None
    closed_at: float | None = None

    def send(self, data: str, event: str | None = None) -> bool:
        if data == CANCELLED_PART:
            self.cancelled_at = time.monotonic()
        return super().send(data, event)

    async def close(self, outcome: Any = "end", timeout: float = 10.0) -> Any:
        if self.closed_at is None:
            self.closed_at = time.monotonic()
        return await super().close(outcome, timeout)


@pytest.mark.parametrize("stop", ["break", "cancel"])
async def test_an_async_cancel_is_published_before_the_graph_has_stopped(stop: str) -> None:
    """A tool that finishes its work before it lets a cancel through holds LangGraph up, whether the loop was left or
    the task cancelled while LangGraph waited for the tool; the page hears at once, and the close line is fail."""
    publisher = AsyncTimedPublisher()
    entered = asyncio.Event()

    async def run() -> None:
        relayed = arelay(publisher, stubborn_agent(1.0, entered), USER, {"configurable": {"thread_id": "slow"}})
        async with publisher, contextlib.aclosing(relayed):
            async for part in relayed:
                if stop == "break" and part["type"] == "tools":
                    await entered.wait()
                    break

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    await asyncio.sleep(0.1)
    stopped = time.monotonic()
    if stop == "cancel":
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert publisher.cancelled_at is not None and publisher.cancelled_at - stopped < 0.5
    assert publisher.closed_at is not None and publisher.closed_at - stopped < 0.5
    assert time.monotonic() - stopped > 0.8, "the graph stopped once the tool was done"
    assert publisher.data.count(CANCELLED_PART) == 1
    assert publisher.outcome == "fail"


async def test_with_close_on_cancel_end_the_close_line_waits_for_the_graph() -> None:
    """end says the answer is complete as it stands: only once the graph has stopped and saved it."""
    publisher = AsyncTimedPublisher()
    entered = asyncio.Event()

    async def run() -> None:
        async with publisher:
            config: Any = {"configurable": {"thread_id": "slow"}}
            await apublish_run(publisher, stubborn_agent(1.0, entered), USER, config, close_on_cancel="end")

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    stopped = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert publisher.cancelled_at is not None and publisher.cancelled_at - stopped < 0.5
    assert publisher.closed_at is not None and publisher.closed_at - stopped > 0.8
    assert publisher.outcome == "end"
