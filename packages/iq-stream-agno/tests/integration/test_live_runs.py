"""Agno runs published through the real IQClient to the two real servers, and what a real subscriber received."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ag_ui.core import Event
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from iq_stream import IQClient
from iq_stream.testing import TestServers, asubscribe, subscribe_in_background
from live import delivered, short_client
from pydantic import TypeAdapter
from scripted_model import BIG, SLOW, weather_agent

from iq_stream_agno import MASKED_ERROR, apublish, publish, start_run
from iq_stream_agno._native import dumps

QUIET = ("ModelRequestStarted", "ModelRequestCompleted")
EVENTS: TypeAdapter[Any] = TypeAdapter(Event)


async def test_an_async_run_with_a_tool_call_arrives_in_order_once_and_ends(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    publisher = iq_client.apublisher(ticket.stream_id)
    received = [
        event async for event in apublish(weather_agent().arun("Weather?", stream=True, stream_events=True), publisher)
    ]
    rows = delivered(await subscription, "iq:end")
    assert rows == [json.loads(dumps(event.to_dict())) for event in received]
    assert [row["event"] for row in rows][-1] == "RunCompleted" and "ToolCallCompleted" in [
        row["event"] for row in rows
    ]
    assert publisher.requests == 1


def test_a_sync_run_in_ag_ui_arrives_valid_and_ends(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with iq_client.publisher(ticket.stream_id) as publisher:
        for _ in publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher, protocol="agui"):
            pass
    rows = delivered(subscription.result(10), "iq:end")
    for frame in subscription.result().frames:
        EVENTS.validate_json(frame.data)
    kinds = [row["type"] for row in rows]
    assert kinds[0] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED"
    assert {"TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT"} <= set(kinds)


async def test_a_failing_run_ends_with_its_error_frame_then_iq_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    stream = weather_agent(fail_on_turn=2).arun("Weather?", stream=True, stream_events=True)
    async for _ in apublish(stream, iq_client.apublisher(ticket.stream_id)):
        pass
    result = await subscription
    rows = delivered(result, "iq:fail")
    assert rows[-1]["event"] == "RunError" and rows[-1]["content"] == MASKED_ERROR
    assert result.terminal is not None and result.terminal.data["reason"] == "publisher"


async def test_a_tool_silent_past_the_read_timeout_stays_on_one_request(iq_servers: TestServers) -> None:
    SLOW["seconds"] = 3.0  # the short server answers 408 after 2 s of silence
    client = short_client(iq_servers, keepalive=0.5)
    ticket = client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    publisher = client.apublisher(ticket.stream_id)
    stream = weather_agent(tool_name="slow_weather").arun("Weather?", stream=True, stream_events=True)
    async for _ in apublish(stream, publisher, protocol="agui", exclude=QUIET):
        pass
    rows = delivered(await subscription, "iq:end")
    assert rows[-1]["type"] == "RUN_FINISHED" and publisher.requests == 1


async def test_without_keepalives_the_core_continues_the_stream_after_the_408(iq_servers: TestServers) -> None:
    SLOW["seconds"] = 3.0
    client = short_client(iq_servers, keepalive=None)
    ticket = client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    publisher = client.apublisher(ticket.stream_id)
    stream = weather_agent(tool_name="slow_weather").arun("Weather?", stream=True, stream_events=True)
    async for _ in apublish(stream, publisher, protocol="agui", exclude=QUIET):
        pass
    rows = delivered(await subscription, "iq:end")
    assert rows[-1]["type"] == "RUN_FINISHED" and publisher.requests >= 2


async def test_a_cancelled_run_ends_with_run_cancelled_then_iq_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    SLOW["seconds"] = 0.5
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    stream = weather_agent(tool_name="slow_weather").arun(
        "Weather?", stream=True, stream_events=True, run_id=ticket.stream_id
    )
    async for event in apublish(stream, iq_client.apublisher(ticket.stream_id)):
        if event.event == "ToolCallStarted":
            Agent.cancel_run(ticket.stream_id)
    rows = delivered(await subscription, "iq:fail")
    assert "RunCancelled" in [row["event"] for row in rows]


async def test_a_frame_over_the_short_server_s_limit_is_shrunk_and_delivered(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    BIG["size"] = 3_000
    ticket = iq_short_client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    publisher = iq_short_client.apublisher(ticket.stream_id)
    stream = weather_agent(tool_name="big_report").arun("Report?", stream=True, stream_events=True)
    include = {"RunStarted", "ToolCallStarted", "ToolCallCompleted", "RunCompleted"}
    async for _ in apublish(stream, publisher, include=include):
        pass
    result = await subscription
    rows = delivered(result, "iq:end")
    assert [row["event"] for row in rows] == ["RunStarted", "ToolCallStarted", "ToolCallCompleted", "RunCompleted"]
    # At 1,024 bytes the event's own members leave little room: the result is cut first, then the text.
    assert rows[2]["iq_truncated"][0] == "tool.result" and len(rows[2]["tool"]["result"]) < 3_000
    assert all(publisher.frame_size(frame.data) <= 1024 for frame in result.frames)


async def test_a_run_past_the_stream_byte_limit_fails_the_stream_and_still_completes(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    agent = weather_agent(db=InMemoryDb(), answer=" ".join(["sunny"] * 60))
    ticket = iq_short_client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    publisher = iq_short_client.apublisher(ticket.stream_id)
    stream = agent.arun("Weather?", stream=True, stream_events=True, session_id="s-limit", run_id=ticket.stream_id)
    received = [event async for event in apublish(stream, publisher)]
    result = await subscription
    assert result.terminal is not None and result.terminal.event == "iq:fail"
    assert result.terminal.data["reason"] == "too_large" and not publisher.alive
    assert received[-1].event == "RunCompleted"  # the tee went on reading after the stream was closed
    stored: Any = await agent.aget_run_output(ticket.stream_id, session_id="s-limit")
    assert str(getattr(stored.status, "value", stored.status)).lower() == "completed"


async def test_a_paused_run_continues_on_a_second_stream(iq_client: IQClient, iq_servers: TestServers) -> None:
    agent = weather_agent(tool_name="book_trip", answer="Your trip is booked.", db=InMemoryDb())
    first = start_run(iq_client, agent, "Book Paris", session_id="s-live-hitl")
    leg_one = await asubscribe(first.url)
    assert (await first.task).status == "paused"
    assert delivered(leg_one, "iq:end")[-1]["event"] == "RunPaused"
    paused: Any = await agent.aget_run_output(first.run_id, session_id="s-live-hitl")
    for requirement in paused.requirements:
        requirement.confirm()
    from iq_stream_agno import continue_run

    second = continue_run(
        iq_client, agent, run_id=first.run_id, session_id="s-live-hitl", requirements=paused.requirements
    )
    leg_two = await asubscribe(second.url)
    rows = delivered(leg_two, "iq:end")
    assert rows[0]["event"] == "RunContinued" and rows[-1]["event"] == "RunCompleted"
    assert {row["run_id"] for row in rows} == {first.run_id} and second.stream_id != first.stream_id


async def test_a_late_subscriber_gets_iq_gone_and_the_run_is_stored(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    agent = weather_agent(db=InMemoryDb())
    handle = start_run(iq_client, agent, "Weather?", session_id="s-late")
    summary = await handle.task
    late = await asubscribe(handle.url)
    assert late.frames == [] and late.terminal is not None and late.terminal.event == "iq:gone"
    assert late.terminal.data["outcome"] == "end" and late.terminal.data["lastEventId"] == summary.frames
    stored: Any = await agent.aget_run_output(handle.run_id, session_id="s-late")
    assert str(stored.content).endswith("sunny in Paris.")


async def test_a_subscriber_that_resumes_gets_the_rest(iq_client: IQClient, iq_servers: TestServers) -> None:
    SLOW["seconds"] = 1.0
    handle = start_run(iq_client, weather_agent(tool_name="slow_weather"), "Weather?")
    await asyncio.sleep(0.4)  # the frames before the tool are out; the tool takes a second
    resumed = await asubscribe(iq_servers.default.url, handle.stream_id, handle.token, last_event_id=3)
    summary = await handle.task
    assert [int(frame.id) for frame in resumed.frames] == list(range(4, summary.frames + 1))
    assert resumed.terminal is not None and resumed.terminal.event == "iq:end"


async def test_ten_concurrent_runs_of_one_agent_do_not_mix(iq_client: IQClient, iq_servers: TestServers) -> None:
    SLOW["seconds"] = 0.3
    agent = weather_agent(tool_name="slow_weather")
    handles = [start_run(iq_client, agent, "Weather?") for _ in range(10)]
    subscriptions = await asyncio.gather(*(asubscribe(handle.url) for handle in handles))
    await asyncio.gather(*(handle.task for handle in handles))
    for handle, subscription in zip(handles, subscriptions, strict=True):
        rows = delivered(subscription, "iq:end")
        assert {row["run_id"] for row in rows} == {handle.run_id}


def test_start_run_needs_no_extra_step_to_reach_a_browser(iq_servers: TestServers) -> None:
    client = IQClient(iq_servers.default.url, iq_servers.app_key, iq_servers.app_secret)  # default min_open: 3 s

    async def main() -> Any:
        handle = start_run(client, weather_agent(), "Weather?")
        await asyncio.sleep(0.5)  # a browser that subscribes after a quick answer finished
        return handle, await asubscribe(handle.url), await handle.task

    handle, subscription, summary = asyncio.run(main())
    rows = delivered(subscription, "iq:end")
    assert len(rows) == summary.frames and rows[0]["run_id"] == handle.run_id
