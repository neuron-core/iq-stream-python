"""What is particular to each publisher: cancellation and the shield of the async close, the context managers,
threads, the atexit hook, and a synchronous publisher inside a running loop."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fakes import STREAM, Harness, closed_end, closed_fail, until

from iq_stream import AsyncPublisher, IQClient, Publisher
from iq_stream import _publisher as publisher_module
from iq_stream._publisher import close_open_publishers


async def test_the_async_close_is_shielded_from_the_cancellation_of_its_caller() -> None:
    h = Harness("async")
    pub = h.publisher()
    pub.send("a")
    request = await h.request(0)
    caller = asyncio.ensure_future(pub.close("fail"))
    await until(lambda: request.ended)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert request.closes == ["fail"], "the close line went out"
    request.respond(200, lastEventId=1, closed=True, outcome="fail", reason="publisher")
    assert (await pub.close()).outcome == "fail", "and its answer is still the result"


async def test_a_task_cancelled_inside_the_async_context_manager_closes_with_fail_at_once() -> None:
    h = Harness("async")
    h.transport.on_open = lambda request, n: None
    started = asyncio.Event()

    async def run() -> None:
        async with h.client.apublisher(STREAM) as pub:
            pub.send("partial")
            started.set()
            await asyncio.sleep(30)

    task = asyncio.ensure_future(run())
    await started.wait()
    request = await h.request(0)
    task.cancel()
    await until(lambda: request.ended, "the close line", 1.0)
    assert request.closes == ["fail"]
    request.respond(200, lastEventId=1, closed=True, outcome="fail", reason="publisher")
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_keyboard_interrupt_in_the_sync_context_manager_closes_with_fail() -> None:
    h = Harness("sync")
    h.transport.on_post = lambda body, n: closed_fail(1)

    def run() -> None:
        with h.client.publisher(STREAM) as pub:
            pub.send("partial")
            raise KeyboardInterrupt

    async def answer() -> None:
        request = await h.request(0)
        await until(lambda: request.ended)
        request.respond(200, lastEventId=1, closed=True, outcome=request.closes[0], reason="publisher")

    answering = asyncio.ensure_future(answer())
    with pytest.raises(KeyboardInterrupt):
        await asyncio.to_thread(run)
    await answering
    assert h.transport.requests[0].closes == ["fail"]


async def test_a_writer_task_cancelled_with_its_loop_closes_the_stream_out_of_band() -> None:
    h = Harness("async")
    pub = h.publisher()
    pub.send("a")
    await h.request(0)
    pub._task.cancel()  # pyright: ignore[reportPrivateUsage] - as asyncio.run() does at its end
    await until(lambda: pub.error is not None)
    assert pub.error.code == "cancelled"
    assert (await pub.close()).outcome == "fail"
    assert h.transport.closes == ["fail"]


async def test_a_writer_task_cancelled_after_a_close_was_asked_for_sends_that_close() -> None:
    h = Harness("async", min_open=5)
    h.transport.on_post = lambda body, n: closed_fail(1) if "fail" in body else closed_end(1)
    pub = h.publisher()
    pub.send("a")
    await h.request(0)
    closing = asyncio.ensure_future(pub.close("end"))
    await asyncio.sleep(0.05)
    pub._task.cancel()  # pyright: ignore[reportPrivateUsage]
    outcome = await closing
    assert h.transport.closes == ["end"]
    assert (outcome.closed, outcome.outcome, pub.error) == (True, "end", None)


async def test_the_atexit_hook_closes_every_open_sync_publisher_with_fail() -> None:
    h = Harness("sync")
    h.transport.on_open = lambda request, n: None
    first = h.client.publisher(STREAM)
    second = h.client.publisher("another-stream-0000001")
    first.send("a")

    async def answer() -> None:
        for n in range(2):
            request = await h.request(n)
            await until(lambda request=request: request.ended)
            request.respond(200, lastEventId=0, closed=True, outcome=request.closes[0], reason="publisher")

    answering = asyncio.ensure_future(answer())
    assert first in publisher_module._open  # pyright: ignore[reportPrivateUsage]
    await asyncio.to_thread(close_open_publishers)
    await answering
    assert [request.closes for request in h.transport.requests] == [["fail"], ["fail"]]
    assert first.alive is second.alive is False
    assert first not in publisher_module._open  # pyright: ignore[reportPrivateUsage]
    assert publisher_module._hooked  # pyright: ignore[reportPrivateUsage]


async def test_a_sync_publisher_works_inside_a_running_loop() -> None:
    h = Harness("sync")
    pub = h.client.publisher(STREAM)
    assert isinstance(pub, Publisher)
    for i in range(3):
        assert pub.send(f"f{i}") is True  # from the loop's thread, without blocking it
    request = await h.request(0)
    await until(lambda: len(request.data) == 3)
    closing = asyncio.ensure_future(asyncio.to_thread(pub.close))
    await until(lambda: request.ended)
    request.respond(200, lastEventId=3, closed=True, outcome="end")
    assert (await closing).last_event_id == 3


async def test_close_may_come_from_another_thread_than_the_ones_that_created_and_sent() -> None:
    h = Harness("sync")
    pub = h.client.publisher(STREAM)
    senders = [threading.Thread(target=lambda i=i: [pub.send(f"{i}-{j}") for j in range(50)]) for i in range(4)]
    for sender in senders:
        sender.start()
    for sender in senders:
        sender.join()
    request = await h.request(0)
    await until(lambda: len(request.data) == 200)
    for i in range(4):
        mine = [data for data in request.data if data.startswith(f"{i}-")]
        assert mine == [f"{i}-{j}" for j in range(50)], "each thread's frames in the order its calls returned"
    result: list[object] = []
    closer = threading.Thread(target=lambda: result.append(pub.close()))
    closer.start()
    await until(lambda: request.ended)
    request.respond(200, lastEventId=200, closed=True, outcome="end")
    await asyncio.to_thread(closer.join)
    assert getattr(result[0], "last_event_id", None) == 200


def test_apublisher_needs_a_running_loop() -> None:
    client = IQClient("http://iq.test", "demo-app", "x" * 32)
    with pytest.raises(RuntimeError, match="running event loop"):
        client.apublisher(STREAM)


async def test_apublisher_returns_an_async_publisher_whose_request_opens_when_the_loop_runs() -> None:
    h = Harness("async")
    pub = h.publisher()
    assert isinstance(pub, AsyncPublisher)
    assert h.transport.requests == [], "not before the loop runs"
    request = await h.request(0)
    request.respond(200, lastEventId=0, closed=True, outcome="end")
    assert (await pub.close()).closed is True
