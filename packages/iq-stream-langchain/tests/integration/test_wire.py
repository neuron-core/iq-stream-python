"""Runs published through the real IQClient to the real servers, as a subscriber received them."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fakes import (
    ASYNC_TOKENS,
    FINAL,
    FIRST,
    TOOL_RESULT,
    USER,
    FakeToolCallingModel,
    failing_agent,
    get_weather,
    text_of,
    tool_call,
    weather_agent,
)
from iq_stream import IQClient, IQError
from iq_stream.testing import TestServers, free_port
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from streams import parts, watch

from iq_stream_langchain import CANCELLED_PART, MASKED_MESSAGE, Oversize, apublish_run, arelay, publish_run, relay

TYPES = [
    "metadata",
    *["messages"] * 9,
    "updates",
    "tools",
    "tools",
    "messages",
    "updates",
    *["messages"] * 8,
    "updates",
]


def compact_agent(tool: Any = get_weather, *, delay: float = 0.0) -> Any:
    """One tool call and a one-word answer: small enough for the short server's 8,192-byte streams."""
    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call(tool.name, {"city": "Turin"}, "c1")]), "Done."]),  # type: ignore[list-item]
        token_delay=delay,
    )
    return create_agent(model, [tool])


def test_a_sync_run_with_a_tool_call_arrives_in_order_on_one_request(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket, subscription = watch(iq_client, iq_servers.default.url)
    with iq_client.publisher(ticket.stream_id) as publisher:
        yielded = [part["type"] for part in relay(publisher, weather_agent(), USER)]
    received = subscription.result(10)

    assert received.terminal is not None and received.terminal.event == "iq:end"
    published = parts(received)
    assert [part["type"] for part in published] == TYPES == ["metadata", *yielded]
    assert text_of(published) == FIRST + FINAL
    assert [part["data"]["event"] for part in published if part["type"] == "tools"] == ["tool-started", "tool-finished"]
    assert any(part["type"] == "messages" and part["data"][0]["content"] == TOOL_RESULT for part in published)
    assert publisher.requests == 1 and publisher.error is None


async def test_an_async_run_arrives_in_order_on_one_request(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket, subscription = watch(iq_client, iq_servers.default.url)
    async with iq_client.apublisher(ticket.stream_id) as publisher:
        yielded = [part["type"] async for part in arelay(publisher, weather_agent(), USER)]
    received = await subscription

    assert received.terminal is not None and received.terminal.event == "iq:end"
    published = parts(received)
    assert [part["type"] for part in published] == ["metadata", *yielded]
    if ASYNC_TOKENS:
        assert yielded == TYPES[1:]
    assert text_of(published) == FIRST + FINAL
    assert publisher.requests == 1 and publisher.error is None


def test_a_failing_run_ends_with_its_error_part_then_iq_fail(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket, subscription = watch(iq_client, iq_servers.default.url)
    with pytest.raises(RuntimeError, match="exploded"), iq_client.publisher(ticket.stream_id) as publisher:
        publish_run(publisher, failing_agent(), USER)
    received = subscription.result(10)

    published = parts(received)
    assert published[-1] == {"type": "error", "ns": [], "data": {"error": "RuntimeError", "message": MASKED_MESSAGE}}
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "publisher"


def test_a_silent_tool_longer_than_the_read_timeout_keeps_one_request(iq_servers: TestServers) -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        time.sleep(3.0)  # the short server's read timeout is 2 s
        return f"sunny, 21C in {city}"

    client = IQClient(
        iq_servers.short.url, iq_servers.app_key, iq_servers.app_secret, min_open=0, max_frame_bytes=1024, keepalive=0.5
    )
    ticket, subscription = watch(client, iq_servers.short.url)
    with client.publisher(ticket.stream_id) as publisher:
        final = publish_run(publisher, compact_agent(get_weather), USER)
    received = subscription.result(10)

    assert final["messages"][-1].content == "Done."
    assert received.terminal is not None and received.terminal.event == "iq:end"
    published = parts(received)
    tools = [
        (part["data"]["event"], frame.at)
        for part, frame in zip(published, received.frames, strict=True)
        if part["type"] == "tools"
    ]
    assert [event for event, _ in tools] == ["tool-started", "tool-finished"]
    assert tools[1][1] - tools[0][1] >= 2.5, "the stream was silent for longer than the read timeout"
    assert publisher.requests == 1, "keepalives held the request open"
    assert publisher.error is None


async def test_a_cancelled_run_ends_with_the_cancel_part_then_iq_fail_at_once(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    tool_started = asyncio.Event()

    @tool
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        tool_started.set()
        await asyncio.sleep(30)
        return "never"

    ticket, subscription = watch(iq_client, iq_servers.default.url)

    async def run() -> None:
        async with iq_client.apublisher(ticket.stream_id) as publisher:
            await apublish_run(publisher, compact_agent(get_weather), USER)

    task = asyncio.create_task(run())
    await asyncio.wait_for(tool_started.wait(), 10)
    cancelled_at = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    received = await subscription

    assert time.monotonic() - cancelled_at < 2.0, "subscribers hear of the cancel at once, not at the idle timeout"
    published = parts(received)
    assert received.frames[-1].data == CANCELLED_PART
    assert [part["data"]["event"] for part in published if part["type"] == "tools"] in ([], ["tool-started"])
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "publisher"


def test_a_part_over_the_short_server_frame_limit_is_cut_by_the_adapter(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "rain " * 600

    reports: list[Oversize] = []
    ticket, subscription = watch(iq_short_client, iq_servers.short.url)
    with iq_short_client.publisher(ticket.stream_id) as publisher:
        final = publish_run(publisher, compact_agent(get_weather), USER, on_oversize=reports.append)
    received = subscription.result(10)

    assert final["messages"][-2].content == "rain " * 600, "the state holds the whole result"
    assert received.terminal is not None and received.terminal.event == "iq:end"
    published = parts(received)
    assert max(len(frame.data.encode()) for frame in received.frames) < 1024
    tool_message = next(part for part in published if part["type"] == "messages" and part["data"][0]["type"] == "tool")
    assert tool_message["data"][0]["content"].startswith("rain rain") and tool_message["data"][0]["content"].endswith(
        "…[truncated]"
    )
    assert tool_message["data"][1]["truncated"] is True
    assert [(report.part_type, report.truncated) for report in reports] == [
        ("tools", False),
        ("messages", True),
        ("updates", False),
    ]
    assert publisher.error is None


async def test_two_runs_at_once_do_not_mix_their_frames(iq_client: IQClient, iq_servers: TestServers) -> None:
    async def run(agent: Any) -> list[dict[str, Any]]:
        ticket, subscription = watch(iq_client, iq_servers.default.url)
        async with iq_client.apublisher(ticket.stream_id) as publisher:
            await apublish_run(publisher, agent, USER)
        received = await subscription
        assert received.terminal is not None and received.terminal.event == "iq:end"
        return parts(received)

    other = create_agent(FakeToolCallingModel(messages=iter(["alpha one two three"]), token_delay=0.01), [])
    first, second = await asyncio.gather(run(weather_agent(token_delay=0.01)), run(other))
    assert text_of(first) == FIRST + FINAL
    assert text_of(second) == "alpha one two three"


def test_an_unreachable_server_leaves_the_run_alone(iq_servers: TestServers) -> None:
    errors: list[IQError] = []
    client = IQClient(
        f"http://127.0.0.1:{free_port()}",
        iq_servers.app_key,
        iq_servers.app_secret,
        min_open=0,
        retry_budget=0.5,
        error_handler=errors.append,
    )
    with client.publisher(client.new_stream_id()) as publisher:
        final = publish_run(publisher, weather_agent(), USER)
    assert final["messages"][-1].content == FINAL
    assert not publisher.alive
    assert [error.code for error in errors] == ["unreachable"]


def test_a_late_subscriber_gets_gone_and_the_answer_is_in_the_checkpointer(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    from iq_stream.testing import subscribe
    from langgraph.checkpoint.memory import InMemorySaver

    agent = weather_agent(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "late"}}
    ticket = iq_client.ticket()
    with iq_client.publisher(ticket.stream_id) as publisher:
        publish_run(publisher, agent, USER, config)
    late = subscribe(iq_servers.default.url, ticket.stream_id, ticket.token, timeout=10)
    assert late.terminal is not None and late.terminal.event == "iq:gone"
    assert agent.get_state(config).values["messages"][-1].content == FINAL


def test_a_subscriber_that_reconnects_mid_run_misses_nothing(iq_client: IQClient, iq_servers: TestServers) -> None:
    from iq_stream.testing import BackgroundSubscription, subscribe_in_background

    ticket, first = watch(iq_client, iq_servers.default.url)
    resumed: BackgroundSubscription | None = None
    with iq_client.publisher(ticket.stream_id) as publisher:
        for index, _ in enumerate(relay(publisher, weather_agent(token_delay=0.02), USER)):
            if index == 5:
                time.sleep(0.3)  # the first frames reach the server
                resumed = subscribe_in_background(
                    iq_servers.default.url, ticket.stream_id, ticket.token, last_event_id=3, timeout=30
                )
    assert resumed is not None
    received, again = first.result(10), resumed.result(10)
    assert received.terminal is not None and again.terminal is not None
    assert again.terminal.event == received.terminal.event == "iq:end"
    assert again.frames[0].id == "4"
    assert again.data == received.data[3:], "no gap and no repeat"


def test_a_stream_over_the_server_byte_limit_fails_and_the_run_completes(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    ticket, subscription = watch(iq_short_client, iq_servers.short.url)  # 8,192 bytes a stream
    seen = 0
    with iq_short_client.publisher(ticket.stream_id) as publisher:
        for _ in relay(publisher, weather_agent(token_delay=0.01), USER):
            seen += 1
    received = subscription.result(10)

    assert seen == len(TYPES) - 1, "the run was read to its end"
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "too_large"
    assert not publisher.alive and publisher.error is not None
