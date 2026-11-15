"""publish() through the real IQClient to the two real servers, asserting on what a real subscriber received."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from checks import agui_events, assert_well_formed, joined
from fakes import (
    ANSWER,
    FIRST_TEXT,
    TOOL_RESULT,
    big_weather,
    hitl_agent,
    slow_weather,
    weather_agent,
)
from iq_stream import IQClient, Outcome, PublishResult
from iq_stream.testing import TestServers, asubscribe, start_server, subscribe_in_background
from live import assert_delivered, named, payloads
from llama_index.core.agent.workflow import ToolCall
from workflows.events import Event, HumanResponseEvent, InputRequiredEvent

from iq_stream_llamaindex import AGUIEncoder, RunOutcome, publish


async def test_a_native_run_with_a_tool_call_arrives_in_order_once_on_one_request_and_ends(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    publisher = iq_client.apublisher(ticket.stream_id)
    outcome = await publish(weather_agent().run(user_msg="Weather in Paris?", run_id=ticket.stream_id), publisher)
    received = await subscription
    assert_delivered(received, "iq:end")
    assert publisher.requests == 1 and outcome.frames == len(received.frames)
    assert outcome.publish_result is not None and outcome.publish_result.last_event_id == len(received.frames)
    names = named(received)
    assert names[0] == "RunStarted" and names[-1] == "StopEvent"
    assert names.index("ToolCall") < names.index("ToolCallResult")
    data = payloads(received)
    text = "".join(str(event["delta"]) for name, event in zip(names, data, strict=True) if name == "AgentStream")
    assert text == FIRST_TEXT + ANSWER
    assert data[names.index("ToolCallResult")]["content"] == TOOL_RESULT
    assert data[-1] == {"result": ANSWER}


async def test_an_ag_ui_run_arrives_as_valid_ag_ui_events_in_order(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    publisher = iq_client.apublisher(ticket.stream_id)
    handler = weather_agent().run(user_msg="Weather in Paris?", run_id=ticket.stream_id)
    await publish(handler, publisher, encoder=AGUIEncoder(thread_id="thread-1"))
    received = await subscription
    assert_delivered(received, "iq:end")
    assert set(named(received)) == {"message"} and publisher.requests == 1
    events = agui_events(received.data)
    assert_well_formed(events)
    assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT + ANSWER
    assert events[0]["runId"] == ticket.stream_id and events[-1]["type"] == "RUN_FINISHED"


@pytest.mark.parametrize("protocol", ["native", "agui"])
async def test_a_failing_run_ends_with_iq_fail_right_after_its_error_frame(
    iq_client: IQClient, iq_servers: TestServers, protocol: str
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    encoder = AGUIEncoder(thread_id="t") if protocol == "agui" else None
    outcome = await publish(
        weather_agent(fail_on_second_call=True).run(user_msg="Weather?"),
        iq_client.apublisher(ticket.stream_id),
        encoder=encoder,
    )
    received = await subscription
    assert_delivered(received, "iq:fail")
    assert received.terminal is not None and received.terminal.data["reason"] == "publisher"
    assert outcome.status == "failed" and outcome.published
    last = payloads(received)[-1]
    if protocol == "agui":
        assert last == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "failed"}
    else:
        assert named(received)[-1] == "WorkflowFailedEvent" and last["message"] == "An error occurred."
    assert not any("sk-hidden" in data for data in received.data)


async def test_a_tool_silent_past_the_read_timeout_keeps_one_request_open_with_keepalives(
    iq_servers: TestServers,
) -> None:
    # The short server answers 408 after 2 s of silence; the tool is silent for 3 s.
    client = IQClient(
        iq_servers.short.url, iq_servers.app_key, iq_servers.app_secret, min_open=0, keepalive=0.5, max_frame_bytes=1024
    )
    ticket = client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    publisher = client.apublisher(ticket.stream_id)
    started = time.monotonic()
    outcome = await publish(weather_agent(tool=slow_weather(3.0)).run(user_msg="Weather?"), publisher)
    assert time.monotonic() - started >= 3.0
    received = await subscription
    assert_delivered(received, "iq:end")
    assert publisher.requests == 1 and outcome.status == "completed"
    call, result = named(received).index("ToolCall"), named(received).index("ToolCallResult")
    assert received.frames[result].at - received.frames[call].at >= 2.8  # relayed live, not at the end


async def test_cancel_run_ends_with_the_cancelled_outcome_and_iq_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            asyncio.get_running_loop().create_task(handler.cancel_run())

    outcome = await publish(
        handler, iq_client.apublisher(ticket.stream_id), encoder=AGUIEncoder(thread_id="t"), on_event=on_event
    )
    received = await subscription
    assert_delivered(received, "iq:fail")
    assert outcome.status == "cancelled"
    assert payloads(received)[-1]["outcome"] == {"type": "cancelled"}


async def test_cancelling_the_publishing_task_closes_the_stream_with_fail_at_once(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    called = asyncio.Event()

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            called.set()

    task = asyncio.create_task(publish(handler, iq_client.apublisher(ticket.stream_id), on_event=on_event))
    await asyncio.wait_for(called.wait(), 5)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    received = await subscription
    assert time.monotonic() - started < 3
    assert_delivered(received, "iq:fail")
    assert named(received)[-1] == "RunAborted" and payloads(received)[-1] == {"reason": "cancelled"}
    assert handler.is_done()


async def test_frames_over_the_servers_1024_byte_limit_are_cut_by_the_adapter_not_refused(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_short_client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    agent = weather_agent(tool=big_weather, answer_tokens=["Here: ", "«sunny» " * 100])
    outcome = await publish(agent.run(user_msg="Weather?"), iq_short_client.apublisher(ticket.stream_id))
    received = await subscription
    assert_delivered(received, "iq:end")
    assert outcome.published and outcome.status == "completed"
    result = payloads(received)[named(received).index("ToolCallResult")]
    assert result["truncated"] is True and str(result["content"]).endswith("…")
    deltas = [
        event["delta"] for name, event in zip(named(received), payloads(received), strict=True) if name == "AgentStream"
    ]
    assert "".join(map(str, deltas)).endswith("Here: " + "«sunny» " * 100)
    assert sum("«sunny»" in str(delta) for delta in deltas) >= 2  # the long delta went out split
    assert all(len(json.dumps({"data": data}, ensure_ascii=False).encode()) <= 1100 for data in received.data)


async def test_a_stream_past_the_servers_byte_limit_fails_open_and_the_answer_is_still_saved(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_short_client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    saved: list[RunOutcome] = []
    tokens = [f"word{n} " for n in range(300)]  # far past the short server's 8,192-byte stream
    outcome = await publish(
        weather_agent(answer_tokens=tokens, token_delay=0).run(user_msg="Weather?"),
        iq_short_client.apublisher(ticket.stream_id),
        before_close=saved.append,
    )
    received = await subscription
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "too_large"
    assert outcome.status == "completed" and str(outcome.result) == "".join(tokens)
    assert not outcome.published and saved and outcome.publish_result is None


async def test_the_answer_is_saved_before_the_close_so_a_late_browser_finds_it(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    store: dict[str, object] = {}

    async def save_answer(outcome: RunOutcome) -> None:
        await asyncio.sleep(0.2)  # a database write
        store.update(answer=str(outcome.result), at=time.monotonic())

    publisher = iq_client.apublisher(ticket.stream_id)
    close = publisher.close

    async def timed_close(outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        store["closing_at"] = time.monotonic()  # the moment the close line is sent
        return await close(outcome, timeout)

    publisher.close = timed_close  # type: ignore[method-assign]
    outcome = await publish(weather_agent().run(user_msg="Weather?"), publisher, before_close=save_answer)
    late = await asubscribe(iq_servers.default.url, ticket.stream_id, ticket.token, timeout=10)
    assert late.terminal is not None and late.terminal.event == "iq:gone"
    assert late.terminal.data["outcome"] == "end" and late.frames == []
    assert store["answer"] == ANSWER and float(store["at"]) < float(store["closing_at"])  # type: ignore[arg-type]
    assert outcome.publish_result is not None and outcome.publish_result.outcome == "end"


async def test_a_subscriber_joining_mid_run_receives_every_frame_from_the_first(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    joined_late: list[object] = []

    async def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            joined_late.append(
                await asyncio.to_thread(subscribe_in_background, iq_servers.default.url, ticket.stream_id, ticket.token)
            )

    outcome = await publish(
        weather_agent(tool=slow_weather(0.5)).run(user_msg="Weather?"),
        iq_client.apublisher(ticket.stream_id),
        on_event=on_event,
    )
    (subscription,) = joined_late
    received = await subscription  # type: ignore[misc]
    assert_delivered(received, "iq:end")
    assert len(received.frames) == outcome.frames and named(received)[0] == "RunStarted"


async def test_a_human_input_pause_holds_one_request_open_across_keepalives_until_the_answer(
    iq_servers: TestServers,
) -> None:
    client = IQClient(
        iq_servers.short.url, iq_servers.app_key, iq_servers.app_secret, min_open=0, keepalive=0.5, max_frame_bytes=1024
    )
    ticket = client.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    handler = hitl_agent().run(user_msg="Do it")
    publisher = client.apublisher(ticket.stream_id)

    def on_event(event: Event) -> None:
        if isinstance(event, InputRequiredEvent):  # the browser answers 3 s later, past the 2 s read timeout
            asyncio.get_running_loop().call_later(3, handler.ctx.send_event, HumanResponseEvent(response="yes"))

    outcome = await publish(handler, publisher, encoder=AGUIEncoder(thread_id="t"), on_event=on_event)
    received = await subscription
    assert_delivered(received, "iq:end")
    assert publisher.requests == 1 and outcome.status == "completed"
    events = agui_events(received.data)
    assert_well_formed(events)
    assert any(event["type"] == "CUSTOM" and event["name"] == "input_required" for event in events)
    assert joined(events, "TEXT_MESSAGE_CONTENT") == "Done."


async def test_concurrent_runs_publish_to_their_own_streams_without_mixing(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    tickets = [iq_client.ticket() for _ in range(20)]
    subscriptions = [subscribe_in_background(iq_servers.default.url, t.stream_id, t.token) for t in tickets]
    agent = weather_agent()
    await asyncio.gather(
        *(
            publish(
                agent.run(user_msg="Weather?", run_id=t.stream_id),
                iq_client.apublisher(t.stream_id),
                encoder=AGUIEncoder(thread_id=t.stream_id),
            )
            for t in tickets
        )
    )
    for ticket, subscription in zip(tickets, subscriptions, strict=True):
        received = await subscription
        assert_delivered(received, "iq:end")
        events = agui_events(received.data)
        assert {event.get("runId") for event in events} - {None} == {ticket.stream_id}
        assert {event.get("threadId") for event in events} - {None} == {ticket.stream_id}
        assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT + ANSWER


async def test_a_server_killed_mid_run_does_not_break_the_run(iq_servers: TestServers) -> None:
    server = start_server()
    try:
        client = IQClient(server.url, iq_servers.app_key, iq_servers.app_secret, min_open=0, retry_budget=1.0)
        saved: list[RunOutcome] = []

        async def on_event(event: Event) -> None:
            if isinstance(event, ToolCall):
                await asyncio.to_thread(server.kill)

        outcome = await publish(
            weather_agent(tool=slow_weather(0.5)).run(user_msg="Weather?"),
            client.apublisher(client.new_stream_id()),
            on_event=on_event,
            before_close=saved.append,
        )
        assert outcome.status == "completed" and str(outcome.result) == ANSWER
        assert saved and not outcome.published
    finally:
        server.stop()
