"""The scenarios of core-client.md ("Verified behaviour") against the real servers with a real subscriber, for
Publisher and for AsyncPublisher (T7 is the asyncio half of every test here). Each asserts on what the subscriber
received: the frames, their order, no duplicates, and the terminal event."""

from __future__ import annotations

import asyncio
import time

import pytest
from helpers import Proxy, Run, until

from iq_stream import FrameTooLarge, IQError, PublishResult
from iq_stream.testing import Terminal, free_port, subscribe


def ended(last_event_id: int) -> PublishResult:
    return PublishResult(200, last_event_id=last_event_id, closed=True, outcome="end")


async def test_t1_one_request_and_each_frame_reaches_the_subscriber_as_it_is_written(run: Run) -> None:
    iq = run.client(run.servers.default, keepalive=0.3)
    stream_id, subscription = await run.listen(iq, run.servers.default)
    pub = run.publisher(iq, stream_id)
    assert pub.send("one") is True
    await asyncio.sleep(0.5)
    pub.send("two")
    await asyncio.sleep(0.5)
    pub.send("three", "token")
    await asyncio.sleep(0.5)
    pub.send("four")
    assert await run.close(pub) == ended(4)
    assert pub.requests == 1

    received = await subscription
    assert received.status == 200
    assert received.data == ["one", "two", "three", "four"]
    assert [frame.event for frame in received.frames] == ["message", "message", "token", "message"]
    assert [frame.id for frame in received.frames] == ["1", "2", "3", "4"]
    assert received.terminal == Terminal("iq:end", {"lastEventId": 4})
    # Sent 500 ms apart: frames held back until a buffer filled or the body ended would arrive together.
    for before, after in zip(received.frames, received.frames[1:], strict=False):
        assert 0.3 < after.at - before.at < 0.7, f"frame {after.id} arrived {after.at - before.at:.3f} s after"
    assert run.errors == []


async def test_the_request_opens_when_the_publisher_is_created_before_any_frame(run: Run) -> None:
    iq = run.client(run.servers.short, keepalive=None)
    stream_id, subscription = await run.listen(iq, run.servers.short)
    pub = run.publisher(iq, stream_id)
    # Nothing is ever written. Only a request whose head reached the server is answered 408 after its read timeout;
    # a stream without a publisher would fail with publisher_timeout instead.
    await until(lambda: pub.requests == 2, "the 408 of a request that sent its head and nothing else", 6)
    assert (await run.close(pub)).last_event_id == 0
    received = await subscription
    assert received.frames == []
    assert received.terminal == Terminal("iq:end", {"lastEventId": 0})
    assert run.errors == []


async def test_t2_silence_past_the_read_timeout_is_seen_without_a_write_and_the_stream_continues(run: Run) -> None:
    iq = run.client(run.servers.short, keepalive=None)
    stream_id, subscription = await run.listen(iq, run.servers.short)
    pub = run.publisher(iq, stream_id)
    pub.send("a")
    pub.send("b")
    await until(lambda: pub.requests == 2, "the request to be reopened, with nothing written since the 408", 6)
    pub.send("c")
    pub.send("d")
    assert await run.close(pub) == ended(4)
    assert pub.requests == 2
    received = await subscription
    assert received.data == ["a", "b", "c", "d"]
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert run.errors == []


async def test_t3_silence_with_keepalives_stays_on_one_request(run: Run) -> None:
    iq = run.client(run.servers.short, keepalive=0.5)
    stream_id, subscription = await run.listen(iq, run.servers.short)
    pub = run.publisher(iq, stream_id)
    pub.send("a")
    await asyncio.sleep(3.2)
    pub.send("b")
    assert (await run.close(pub)).last_event_id == 2
    assert pub.requests == 1
    received = await subscription
    assert received.data == ["a", "b"]
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert run.errors == []


async def test_t4_a_line_over_the_clients_limit_is_refused_before_it_is_sent(run: Run) -> None:
    iq = run.client(run.servers.short, max_frame_bytes=1024)
    stream_id, subscription = await run.listen(iq, run.servers.short)
    pub = run.publisher(iq, stream_id)
    pub.send("x" * (1024 - 11))
    with pytest.raises(FrameTooLarge):
        pub.send("x" * 2000)
    pub.send("after")
    assert (await run.close(pub)).last_event_id == 2
    received = await subscription
    assert [len(text) for text in received.data] == [1013, 5], "a line of exactly the limit is accepted by the server"
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert run.errors == []


async def test_t4_a_line_over_the_servers_limit_is_answered_413_and_the_stream_is_closed_with_fail_at_once(
    run: Run,
) -> None:
    iq = run.client(run.servers.short)
    stream_id, subscription = await run.listen(iq, run.servers.short)
    started = time.monotonic()
    pub = run.publisher(iq, stream_id)
    pub.send("ok")
    pub.send("x" * 2000)
    await until(lambda: not pub.alive, "the publisher to give up")
    assert pub.send("after") is False, "fail open: no exception"
    assert (pub.error.status, pub.error.code) == (413, "frame_too_large")
    assert await run.close(pub) == PublishResult(
        413, last_event_id=1, closed=True, outcome="fail", reason="publisher", error="frame_too_large"
    )
    assert run.errors == [pub.error], "reported once the stream was closed out of band"
    received = await subscription
    assert received.data == ["ok"]
    assert received.terminal == Terminal("iq:fail", {"reason": "publisher", "lastEventId": 1})
    assert time.monotonic() - started < 4, "not after the idle timeout, as publisher_idle"


async def test_t5_a_stream_past_its_byte_limit_is_closed_by_the_server_and_the_publisher_stops(run: Run) -> None:
    iq = run.client(run.servers.short)
    stream_id, subscription = await run.listen(iq, run.servers.short)
    pub = run.publisher(iq, stream_id)
    for i in range(20):
        pub.send(str(i).ljust(900, "y"))
    outcome = await run.close(pub)
    assert (outcome.status, outcome.error) == (413, "stream_too_large")
    assert (outcome.closed, outcome.outcome, outcome.reason) == (True, "fail", "too_large")
    assert pub.error.code == "stream_too_large"
    assert run.errors == [pub.error]
    received = await subscription
    assert 0 < len(received.frames) < 20
    assert len(received.frames) == outcome.last_event_id
    assert [int(text.rstrip("y")) for text in received.data] == list(range(len(received.frames))), "in order"
    assert received.terminal == Terminal("iq:fail", {"reason": "too_large", "lastEventId": outcome.last_event_id})


async def test_t6_a_connection_killed_mid_stream_is_continued_and_every_frame_arrives_once(run: Run) -> None:
    lossy = Proxy(run.servers.short.url)
    try:
        iq = run.client(lossy.url, keepalive=1.0)
        stream_id, subscription = await run.listen(iq, run.servers.short)
        pub = run.publisher(iq, stream_id)
        for i in range(1, 6):
            pub.send(f"f{i}")
        await asyncio.sleep(0.3)
        lossy.kill()
        for i in range(6, 11):
            pub.send(f"f{i}")
            await asyncio.sleep(0.05)
        assert await run.close(pub) == ended(10)
        assert pub.requests == 2
        received = await subscription
        assert received.data == [f"f{i}" for i in range(1, 11)], "all ten, once each, in order"
        assert received.terminal is not None and received.terminal.event == "iq:end"
        assert run.errors == []
    finally:
        lossy.close()


async def test_t8_a_quick_run_is_gone_for_a_late_browser_and_min_open_keeps_it(run: Run) -> None:
    async def quick(min_open: float) -> tuple[PublishResult, object]:
        iq = run.client(run.servers.default, min_open=min_open)
        ticket = iq.ticket()
        pub = run.publisher(iq, ticket.stream_id)
        pub.send("quick ")
        pub.send("answer")
        closing = asyncio.ensure_future(run.close(pub))
        await asyncio.sleep(0.4)
        received = await asyncio.to_thread(subscribe, run.servers.default.url, ticket.stream_id, ticket.token)
        return await closing, received

    outcome, late = await quick(0)
    assert outcome == ended(2)
    assert late.frames == []  # type: ignore[attr-defined]
    assert late.terminal == Terminal("iq:gone", {"outcome": "end", "lastEventId": 2})  # type: ignore[attr-defined]

    outcome, held = await quick(1.5)
    assert outcome == ended(2)
    assert held.data == ["quick ", "answer"]  # type: ignore[attr-defined]
    assert held.terminal == Terminal("iq:end", {"lastEventId": 2})  # type: ignore[attr-defined]
    assert run.errors == []


async def test_a_closed_stream_refuses_the_request_and_the_publisher_goes_inert_without_hanging(run: Run) -> None:
    iq = run.client(run.servers.default)
    stream_id = iq.new_stream_id()
    first = run.publisher(iq, stream_id)
    first.send("the answer")
    assert (await run.close(first)).closed is True

    started = time.monotonic()
    second = run.publisher(iq, stream_id)
    for i in range(10):
        second.send(f"x{i}")
        await asyncio.sleep(0.02)
    await until(lambda: not second.alive, "the refusal to be seen", 3)
    assert second.send("more") is False
    assert await run.close(second) == PublishResult(409, closed=True, outcome="end", error="stream_closed")
    assert time.monotonic() - started < 3
    assert second.requests == 1
    assert second.error.code == "stream_closed"
    assert run.errors == [second.error]


async def test_abort_leaves_the_stream_open_and_close_stream_ends_it_from_elsewhere(run: Run) -> None:
    iq = run.client(run.servers.default)
    stream_id, subscription = await run.listen(iq, run.servers.default)
    pub = run.publisher(iq, stream_id)
    pub.send("a")
    pub.send("b")
    await asyncio.sleep(0.2)
    pub.abort()
    assert await run.close(pub) == PublishResult(0, closed=False)
    if run.kind == "sync":
        assert await asyncio.to_thread(iq.close_stream, stream_id, "end") == ended(2)
        again = await asyncio.to_thread(iq.close_stream, stream_id)
    else:
        assert await iq.aclose_stream(stream_id, "end") == ended(2)
        again = await iq.aclose_stream(stream_id)
    assert again == PublishResult(409, closed=True, outcome="end", error="stream_closed"), "left as it is"
    received = await subscription
    assert received.data == ["a", "b"]
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert run.errors == []


async def test_bad_credentials_are_fatal_and_reported_and_the_run_goes_on(run: Run) -> None:
    iq = run.client(run.servers.default, app_secret="not-the-secret-of-this-app-0123456789")
    pub = run.publisher(iq, iq.new_stream_id())
    pub.send("a")
    outcome = await run.close(pub)
    assert (outcome.status, outcome.closed) == (401, False)
    assert pub.error.status == 401
    assert run.errors == [pub.error]
    with pytest.raises(IQError) as refused:
        if run.kind == "sync":
            iq.reserve(iq.new_stream_id())
        else:
            await iq.areserve(iq.new_stream_id())
    assert refused.value.status == 401


async def test_a_server_that_cannot_be_reached_is_retried_for_retry_budget_then_given_up(run: Run) -> None:
    iq = run.client(f"http://127.0.0.1:{free_port()}", retry_budget=0.5)
    started = time.monotonic()
    pub = run.publisher(iq, iq.new_stream_id())
    pub.send("a")
    assert await run.close(pub) == PublishResult(0, closed=False, error="unreachable")
    assert time.monotonic() - started < 4
    assert pub.error.code == "unreachable"
    assert run.errors == [pub.error]
