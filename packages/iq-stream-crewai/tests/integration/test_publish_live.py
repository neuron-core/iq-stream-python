"""Crews published through the real IQClient to the two real servers, read back by a real subscriber."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from agui_checks import checked, conversation, types
from crew_fakes import ANSWER, TERM, LookupTool, make_crew
from iq_stream import IQClient, Outcome
from iq_stream.testing import Subscription, TestServers, free_port, start_server, subscribe_in_background

from iq_stream_crewai import publish


def assert_delivered_once_in_order(received: Subscription, frames_sent: int) -> None:
    assert [frame.id for frame in received.frames] == [str(number) for number in range(1, frames_sent + 1)]


def short_client(servers: TestServers, **options: object) -> IQClient:
    return IQClient(
        servers.short.url, servers.app_key, servers.app_secret, min_open=0.0, max_frame_bytes=1024, **options
    )  # type: ignore[arg-type]


def test_a_run_with_a_tool_call_arrives_in_order_once_and_ends(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with publish(iq_client, ticket.stream_id) as run:
        result = make_crew(delay=0.01, tool=LookupTool(seconds=0.2)).kickoff()
    received = subscription.result(30)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert_delivered_once_in_order(received, run.frames_sent)
    events = checked(received.data)
    talk = conversation(events)
    assert talk.text == ANSWER == result.raw
    assert [json.loads(args) for args in talk.tool_args.values()] == [{"term": TERM}]
    assert list(talk.tool_results.values()) == [f"value:{TERM}"]
    assert run.result is not None and run.result.outcome == "end"
    assert run.result.last_event_id == run.frames_sent == len(received.frames)
    assert run.publisher is not None and run.publisher.requests == 1
    assert run.errors == []


def test_the_native_protocol_arrives_with_event_names(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with publish(iq_client, ticket.stream_id, protocol="crewai") as run:
        make_crew().kickoff()
    received = subscription.result(30)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert_delivered_once_in_order(received, run.frames_sent)
    names = [frame.event for frame in received.frames]
    assert names[0] == "crew_kickoff_started" and names[-1] == "crew_kickoff_completed"
    assert "tool_usage_finished" in names


def test_a_failing_run_ends_with_run_error_then_iq_fail(iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with pytest.raises(RuntimeError, match="fake provider outage"), publish(iq_client, ticket.stream_id) as run:
        make_crew(fail_on_call=2).kickoff()
    received = subscription.result(30)
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "publisher"
    events = checked(received.data)
    assert events[-1] == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "error"}
    assert_delivered_once_in_order(received, run.frames_sent)
    assert run.result is not None and run.result.outcome == "fail"


def test_a_silent_tool_longer_than_the_read_timeout_stays_on_one_request(iq_servers: TestServers) -> None:
    iq = short_client(iq_servers, keepalive=0.5)
    ticket = iq.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    with publish(iq, ticket.stream_id) as run:
        make_crew(tool=LookupTool(seconds=3.0)).kickoff()  # the short server's read timeout is 2 s
    received = subscription.result(30)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert_delivered_once_in_order(received, run.frames_sent)
    gap = received.frames[types(checked(received.data)).index("TOOL_CALL_RESULT")].at
    assert gap - received.frames[0].at > 2.5  # the tool's silence really was on the wire
    assert run.publisher is not None and run.publisher.requests == 1


def test_without_keepalives_the_core_resumes_after_the_408(iq_servers: TestServers) -> None:
    iq = short_client(iq_servers, keepalive=None)
    ticket = iq.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    with publish(iq, ticket.stream_id) as run:
        make_crew(tool=LookupTool(seconds=3.0)).kickoff()
    received = subscription.result(30)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert_delivered_once_in_order(received, run.frames_sent)
    assert conversation(checked(received.data)).text == ANSWER
    assert run.publisher is not None and run.publisher.requests >= 2


@pytest.mark.parametrize("close_on_cancel", ["fail", "end"])
async def test_a_cancelled_run(close_on_cancel: Outcome, iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    gate = threading.Event()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    class Signalling(LookupTool):
        def _run(self, term: str = "default") -> str:
            loop.call_soon_threadsafe(started.set)
            return super()._run(term)

    async def run() -> None:
        async with publish(iq_client, ticket.stream_id, close_on_cancel=close_on_cancel):
            await make_crew(tool=Signalling(gate=gate, seconds=10.0)).akickoff()

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    received = await subscription.aresult(30)
    events = checked(received.data)
    assert events[-1]["outcome"] == {"type": "cancelled"}
    assert received.terminal is not None
    if close_on_cancel == "fail":
        assert received.terminal.event == "iq:fail" and received.terminal.data["reason"] == "publisher"
    else:
        assert received.terminal.event == "iq:end"


def test_a_frame_over_the_short_server_s_limit_is_cut(iq_servers: TestServers) -> None:
    iq = short_client(iq_servers, keepalive=0.5)
    ticket = iq.ticket()
    subscription = subscribe_in_background(iq_servers.short.url, ticket.stream_id, ticket.token)
    with publish(iq, ticket.stream_id) as run:
        make_crew(tool=LookupTool(result="r" * 5000)).kickoff()
    received = subscription.result(30)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert_delivered_once_in_order(received, run.frames_sent)
    [result] = conversation(checked(received.data)).tool_results.values()
    assert result.startswith("rrr") and result.endswith("bytes truncated]")
    assert run.errors == []


def test_two_crews_at_once_make_two_clean_streams(iq_client: IQClient, iq_servers: TestServers) -> None:
    answers = ["ALPHA alpha alpha alpha.", "BETA beta beta beta."]
    tickets = [iq_client.ticket() for _ in answers]
    subscriptions = [
        subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token) for ticket in tickets
    ]
    runs = [publish(iq_client, ticket.stream_id) for ticket in tickets]

    def work(index: int) -> None:
        with runs[index]:
            make_crew(name=str(index), answer=answers[index], terms=[answers[index]], delay=0.005).kickoff()

    threads = [threading.Thread(target=work, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    for subscription, run, answer in zip(subscriptions, runs, answers, strict=True):
        received = subscription.result(30)
        assert received.terminal is not None and received.terminal.event == "iq:end"
        assert_delivered_once_in_order(received, run.frames_sent)
        talk = conversation(checked(received.data))
        assert talk.text == answer
        assert list(talk.tool_results.values()) == [f"value:{answer}"]


def test_an_unreachable_server_never_breaks_the_run(iq_servers: TestServers) -> None:
    iq = IQClient(
        f"http://127.0.0.1:{free_port()}", iq_servers.app_key, iq_servers.app_secret, min_open=0.0, retry_budget=1.0
    )
    started = time.monotonic()
    with publish(iq) as run:
        result = make_crew().kickoff()
    assert result.raw == ANSWER
    assert time.monotonic() - started < 15
    assert run.publisher is not None and not run.publisher.alive
    assert run.publisher.error is not None and run.publisher.error.code == "unreachable"
    assert run.errors == []


def test_a_run_past_the_stream_s_duration_limit_finishes_unpublished(iq_servers: TestServers) -> None:
    credentials = {"app_key": iq_servers.app_key, "app_secret": iq_servers.app_secret}
    with start_server(env={"MAX_STREAM_DURATION_MS": "1500"}, **credentials) as server:
        iq = IQClient(server.url, **credentials, min_open=0.0)
        ticket = iq.ticket()
        subscription = subscribe_in_background(server.url, ticket.stream_id, ticket.token)
        with publish(iq, ticket.stream_id) as run:
            result = make_crew(tool=LookupTool(seconds=3.0)).kickoff()
        received = subscription.result(30)
    assert result.raw == ANSWER
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "max_duration"
    assert "TEXT_MESSAGE_START" not in types([json.loads(data) for data in received.data])
    assert run.publisher is not None and not run.publisher.alive
    assert run.errors == []
