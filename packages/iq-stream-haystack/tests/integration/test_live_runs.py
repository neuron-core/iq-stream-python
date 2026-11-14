"""Haystack runs published through the real IQClient to the two real servers, read by a real subscriber."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from haystack.components.agents import Agent
from iq_stream import IQClient
from iq_stream.testing import TestServers, free_port, subscribe, subscribe_in_background

from haystack_integrations.iq_stream import IQStream
from tests.checks import verify_agui, verify_ai_sdk
from tests.fakes import ScriptedChatGenerator, text_turn, tool_turn, weather_call
from tests.live import QUESTION, async_weather_tool, failing_agent, in_order_once, types, weather_agent, weather_tool


def test_a_run_with_a_tool_call_arrives_in_order_once_and_ends(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with IQStream(iq_client, ticket.stream_id) as stream:
        result = weather_agent(tool_seconds=0.3).run(QUESTION, streaming_callback=stream.callback)
    received = subscription.result(30)
    data = in_order_once(received)
    conversation = verify_agui(data)
    assert conversation.texts == ["Let me check.", "It is sunny in Paris."]
    assert conversation.results == {"call_1": "Sunny, 21C in Paris"}
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert received.terminal.data == {"lastEventId": len(data)}
    assert stream.publisher is not None and stream.publisher.requests == 1  # pyright: ignore[reportAttributeAccessIssue]
    assert stream.result is not None and stream.result.last_event_id == len(data)
    assert result["last_message"].text == "It is sunny in Paris."
    # Relayed while the run went on: the call's end reached the browser before the tool had answered.
    by_type = {json.loads(frame.data)["type"]: frame.at for frame in received.frames}
    assert by_type["TOOL_CALL_RESULT"] - by_type["TOOL_CALL_END"] >= 0.2


async def test_an_async_run_arrives_as_ai_sdk_chunks(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    agent = Agent(chat_generator=ScriptedChatGenerator(tool_turn_and_answer(), delay=0.001), tools=[weather_tool()])
    async with IQStream(iq_client, ticket.stream_id, protocol="ai-sdk") as stream:
        await agent.run_async(QUESTION, streaming_callback=stream.acallback)
    received = await subscription
    message = verify_ai_sdk(in_order_once(received))
    assert message.texts == ["Let me check.", "It is sunny in Paris."]
    assert message.tools["call_1"]["output"] == "Sunny, 21C in Paris"
    assert received.terminal is not None and received.terminal.event == "iq:end"


def tool_turn_and_answer() -> list[object]:
    return [
        tool_turn([weather_call()], text=["Let ", "me ", "check."]),
        text_turn(["It ", "is ", "sunny ", "in ", "Paris."]),
    ]


def test_a_failing_run_sends_its_error_frame_then_iq_fail(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with pytest.raises(RuntimeError, match="connection broke"), IQStream(iq_client, ticket.stream_id) as stream:
        failing_agent().run(QUESTION, streaming_callback=stream.callback)
    received = subscription.result(30)
    conversation = verify_agui(in_order_once(received))
    assert conversation.terminal == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "error"}
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "publisher"
    assert received.terminal.data["lastEventId"] == len(received.frames)


def test_a_tool_silent_past_the_read_timeout_keeps_one_request(iq_servers: TestServers) -> None:
    # The short server answers 408 after 2 s of silence; the core's keepalives cover the 4.5 s tool.
    client = IQClient(iq_servers.short.url, iq_servers.app_key, iq_servers.app_secret, min_open=0, keepalive=0.5,
                      max_frame_bytes=1024)  # fmt: skip
    ticket = client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    with IQStream(client, ticket.stream_id) as stream:
        weather_agent(tool_seconds=4.5).run(QUESTION, streaming_callback=stream.callback)
    received = subscription.result(30)
    conversation = verify_agui(in_order_once(received))
    assert conversation.results == {"call_1": "Sunny, 21C in Paris"}
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert stream.publisher is not None and stream.publisher.requests == 1  # pyright: ignore[reportAttributeAccessIssue]
    assert not stream.failed


@pytest.mark.parametrize(("cancelled_close", "terminal"), [("fail", "iq:fail"), ("end", "iq:end")])
async def test_a_cancelled_run_tells_the_browser(
    iq_client: IQClient, iq_servers: TestServers, cancelled_close: str, terminal: str
) -> None:
    started = asyncio.Event()
    agent = Agent(chat_generator=ScriptedChatGenerator(tool_turn_and_answer()), tools=[async_weather_tool(30, started)])
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)

    async def run() -> None:
        options = {"cancelled_close": cancelled_close}
        async with IQStream(iq_client, ticket.stream_id, **options) as stream:  # pyright: ignore[reportArgumentType]
            await agent.run_async(QUESTION, streaming_callback=stream.acallback)

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    received = await subscription
    conversation = verify_agui(in_order_once(received))
    assert conversation.terminal["outcome"] == {"type": "cancelled"}
    assert received.terminal is not None and received.terminal.event == terminal


def test_a_result_over_the_short_server_s_frame_limit_is_truncated_to_fit(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_short_client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    with IQStream(iq_short_client, ticket.stream_id) as stream:
        weather_agent(result="€" * 3_000).run(QUESTION, streaming_callback=stream.callback)
    received = subscription.result(30)
    conversation = verify_agui(in_order_once(received))
    assert conversation.results["call_1"].endswith("\n[truncated, 3000 characters in total]")
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert not stream.failed


def test_two_runs_in_two_threads_do_not_mix(iq_client: IQClient, iq_servers: TestServers) -> None:
    tickets = {city: iq_client.ticket() for city in ("Paris", "Rome")}
    subscriptions = {
        city: subscribe_in_background(iq_servers.default.url, t.stream_id, t.token) for city, t in tickets.items()
    }
    barrier = threading.Barrier(2)

    def run(city: str) -> None:
        agent = weather_agent(tool_seconds=0.2, city=city)
        with IQStream(iq_client, tickets[city].stream_id) as stream:
            barrier.wait(5)
            agent.run(QUESTION, streaming_callback=stream.callback)

    threads = [threading.Thread(target=run, args=(city,)) for city in tickets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    for city, subscription in subscriptions.items():
        received = subscription.result(30)
        conversation = verify_agui(in_order_once(received))
        assert conversation.texts[-1] == f"It is sunny in {city}."
        assert conversation.results == {"call_1": f"Sunny, 21C in {city}"}
        assert all(("Rome" if city == "Paris" else "Paris") not in item for item in received.data)


async def test_two_runs_in_two_tasks_do_not_mix(iq_client: IQClient, iq_servers: TestServers) -> None:
    async def run(city: str) -> tuple[str, list[str]]:
        ticket = iq_client.ticket()
        subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
        async with IQStream(iq_client, ticket.stream_id, protocol="ai-sdk") as stream:
            await weather_agent(city=city).run_async(QUESTION, streaming_callback=stream.acallback)
        received = await subscription
        return city, in_order_once(received)

    for city, data in await asyncio.gather(run("Paris"), run("Rome")):
        assert verify_ai_sdk(data).texts[-1] == f"It is sunny in {city}."
        assert all(("Rome" if city == "Paris" else "Paris") not in item for item in data)


def test_a_late_subscriber_gets_iq_gone_once_the_answer_is_saved(iq_client: IQClient, iq_servers: TestServers) -> None:
    saved: dict[str, object] = {}
    ticket = iq_client.ticket()
    with IQStream(iq_client, ticket.stream_id) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
        saved[ticket.stream_id] = [message.to_dict() for message in result["messages"]]
    received = subscribe(iq_servers.default.url, ticket.stream_id, ticket.token, timeout=10)
    assert received.terminal is not None and received.terminal.event == "iq:gone"
    assert ticket.stream_id in saved  # what the page loads instead


def test_an_unreachable_server_never_breaks_the_run() -> None:
    client = IQClient(f"http://127.0.0.1:{free_port()}", "demo-app", "test-app-secret-0123456789abcdef0123",
                      retry_budget=0.5, min_open=0)  # fmt: skip
    with IQStream(client) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    assert result["last_message"].text == "It is sunny in Paris."
    assert stream.failed and stream.error is not None and stream.error.code == "unreachable"  # pyright: ignore[reportAttributeAccessIssue]


def test_frames_on_the_wire_match_the_types_of_a_scripted_run(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with IQStream(iq_client, ticket.stream_id, protocol="haystack") as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
    received = subscription.result(30)
    in_order_once(received)
    sequence = types(received)
    assert sequence[0] == "start" and sequence[-1] == "finish"
    assert sequence.count("chunk") == len(sequence) - 2
