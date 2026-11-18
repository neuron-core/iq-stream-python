"""What surrounds a stream: reserving it, closing it from elsewhere, a server that restarts under a publisher, a
process that exits with a publisher open, cancellation, and handing the ticket to a browser."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from helpers import Run, until

from iq_stream import IQClient, IQError, PublishResult
from iq_stream.testing import StartedServer, Terminal, TestServers, start_server, subscribe, subscribe_in_background


async def test_reserve_makes_a_stream_wait_the_idle_timeout_for_its_publisher_not_the_pending_timeout(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    iq = iq_short_client
    stream_id = iq.new_stream_id()
    assert iq.reserve(stream_id) == PublishResult(200, last_event_id=0, closed=False)
    subscription = subscribe_in_background(iq_servers.short.url, stream_id, iq.token("sub", stream_id))
    # Well past PENDING_TIMEOUT_MS (5 s). The idle timeout (6 s) counts from the last request, so a second
    # reservation on the way keeps the stream for a queue that is slower still.
    await asyncio.sleep(3.2)
    assert await iq.areserve(stream_id) == PublishResult(200, last_event_id=0, closed=False)
    await asyncio.sleep(3.2)
    async with iq.apublisher(stream_id) as pub:
        pub.send("from the queue")
    received = await subscription
    assert received.data == ["from the queue"]
    assert received.terminal is not None and received.terminal.event == "iq:end"


async def test_a_stream_not_reserved_fails_with_publisher_timeout_and_reserve_refuses_a_closed_stream(
    iq_short_client: IQClient, iq_servers: TestServers
) -> None:
    iq = iq_short_client
    stream_id = iq.new_stream_id()
    received = await asyncio.to_thread(subscribe, iq_servers.short.url, stream_id, iq.token("sub", stream_id))
    assert received.terminal == Terminal("iq:fail", {"reason": "publisher_timeout", "lastEventId": 0})
    with pytest.raises(IQError) as refused:
        iq.reserve(stream_id)
    assert (refused.value.status, refused.value.code) == (409, "stream_closed")


async def test_close_stream_tells_subscribers_at_once_from_a_process_that_never_had_the_publisher(
    run: Run,
) -> None:
    iq = run.client(run.servers.default)
    stream_id, subscription = await run.listen(iq, run.servers.default)
    started = time.monotonic()
    if run.kind == "sync":
        iq.reserve(stream_id)
        outcome = iq.close_stream(stream_id)
    else:
        await iq.areserve(stream_id)
        outcome = await iq.aclose_stream(stream_id)
    assert outcome == PublishResult(200, last_event_id=0, closed=True, outcome="fail", reason="publisher")
    received = await subscription
    assert received.terminal == Terminal("iq:fail", {"reason": "publisher", "lastEventId": 0})
    assert time.monotonic() - started < 1


@pytest.fixture
def own_server(iq_servers: TestServers) -> Iterator[StartedServer]:
    """A server in a container of the test's own, which it may kill and restart."""
    try:
        server = start_server(app_key=iq_servers.app_key, app_secret=iq_servers.app_secret)
    except RuntimeError as error:
        pytest.skip(f"no server container could be started: {error}")
    with server:
        yield server


async def test_a_server_that_restarts_mid_stream_ends_the_publisher_with_fail_and_nothing_is_published_after(
    run: Run, own_server: StartedServer
) -> None:
    server = own_server
    iq = run.client(server.url, replay_frames=2)
    stream_id = iq.new_stream_id()
    before = await asyncio.to_thread(subscribe_in_background, server.url, stream_id, iq.token("sub", stream_id))
    pub = run.publisher(iq, stream_id)
    for i in range(1, 6):
        pub.send(f"f{i}")
    await asyncio.sleep(0.3)
    await asyncio.to_thread(server.kill)
    pub.send("f6")
    await asyncio.to_thread(server.start)
    await until(lambda: not pub.alive, "the publisher to give up", 12)
    assert pub.send("f7") is False
    assert pub.error.code == "gap"
    outcome = await run.close(pub, "end")
    assert (outcome.outcome, outcome.closed) == ("fail", True)
    assert run.errors == [pub.error]

    old = await before
    assert old.data == ["f1", "f2", "f3", "f4", "f5"]
    assert old.terminal is None, "the old server died without a word"
    fresh = await asyncio.to_thread(subscribe, server.url, stream_id, iq.token("sub", stream_id))
    assert fresh.frames == [], "nothing was published into the new stream"
    assert fresh.terminal is not None and fresh.terminal.event == "iq:fail"
    assert fresh.terminal.data["reason"] == "publisher"
    resumed = await asyncio.to_thread(subscribe, server.url, stream_id, iq.token("sub", stream_id), last_event_id=5)
    assert resumed.frames == []
    assert resumed.terminal is not None and resumed.terminal.event == "iq:gone", "told to load the result"


async def test_after_a_restart_an_answer_the_replay_ring_holds_whole_is_published_again_from_its_first_frame(
    run: Run, own_server: StartedServer
) -> None:
    server = own_server
    iq = run.client(server.url)
    stream_id = iq.new_stream_id()
    pub = run.publisher(iq, stream_id)
    for i in range(1, 6):
        pub.send(f"f{i}")
    await asyncio.sleep(0.3)
    await asyncio.to_thread(server.kill)
    pub.send("f6")
    await asyncio.to_thread(server.start)
    subscription = await asyncio.to_thread(subscribe_in_background, server.url, stream_id, iq.token("sub", stream_id))
    pub.send("f7")
    await asyncio.sleep(0.5)
    assert await run.close(pub, "end", timeout=15) == PublishResult(200, last_event_id=7, closed=True, outcome="end")
    received = await subscription
    assert received.data == [f"f{i}" for i in range(1, 8)], "the same frames under the same ids"
    assert [frame.id for frame in received.frames] == [str(i) for i in range(1, 8)]
    assert received.terminal is not None and received.terminal.event == "iq:end"
    assert run.errors == []


async def test_a_process_that_exits_with_a_publisher_open_closes_it_with_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    stream_id = iq_client.new_stream_id()
    subscription = await asyncio.to_thread(
        subscribe_in_background, iq_servers.default.url, stream_id, iq_client.token("sub", stream_id), timeout=15
    )
    script = (
        "import sys\n"
        "from iq_stream import IQClient\n"
        "pub = IQClient(min_open=0.5).publisher(sys.argv[1])\n"
        "pub.send('a')\n"
        "pub.send('b')\n"
    )
    env = {
        **os.environ,
        "IQ_STREAM_URL": iq_servers.default.url,
        "IQ_STREAM_APP_KEY": iq_servers.app_key,
        "IQ_STREAM_APP_SECRET": iq_servers.app_secret,
    }
    started = time.monotonic()
    process = await asyncio.to_thread(
        subprocess.run, [sys.executable, "-c", script, stream_id], env=env, timeout=15, capture_output=True
    )
    assert process.returncode == 0, process.stderr
    assert time.monotonic() - started < 6, "the daemon writer does not keep the process alive"
    received = await subscription
    assert received.data == ["a", "b"]
    assert received.terminal == Terminal("iq:fail", {"reason": "publisher", "lastEventId": 2}), "not publisher_idle"


LOOP_SHUTDOWN = """
import asyncio, sys
from iq_stream import IQClient

async def main():
    pub = IQClient(min_open=float(sys.argv[2])).apublisher(sys.argv[1])
    await asyncio.sleep(0)  # the writer has started, and is still connecting
    for i in range(3):
        pub.send(f"f{i}")
    await asyncio.sleep(float(sys.argv[3]))
    asyncio.get_running_loop().create_task(pub.close("end"))
    await asyncio.sleep(0)
    # main returns while the close waits: asyncio.run() cancels the tasks left, the writer's among them

asyncio.run(main())
"""


@pytest.mark.parametrize(
    ("min_open", "wait", "terminal"),
    [
        (0.0, 0.0, Terminal("iq:fail", {"reason": "publisher", "lastEventId": 0})),
        (5.0, 1.0, Terminal("iq:end", {"lastEventId": 3})),
    ],
    ids=["frames-still-queued", "frames-delivered"],
)
async def test_a_loop_that_shuts_down_while_close_end_waits_ends_the_stream_with_end_only_if_nothing_is_missing(
    iq_client: IQClient, iq_servers: TestServers, min_open: float, wait: float, terminal: Terminal
) -> None:
    stream_id = iq_client.new_stream_id()
    subscription = await asyncio.to_thread(
        subscribe_in_background, iq_servers.default.url, stream_id, iq_client.token("sub", stream_id), timeout=15
    )
    env = {
        **os.environ,
        "IQ_STREAM_URL": iq_servers.default.url,
        "IQ_STREAM_APP_KEY": iq_servers.app_key,
        "IQ_STREAM_APP_SECRET": iq_servers.app_secret,
    }
    process = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", LOOP_SHUTDOWN, stream_id, str(min_open), str(wait)],
        env=env,
        timeout=15,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    received = await subscription
    assert received.terminal == terminal, "never `end` on an answer with a hole in it"
    assert received.data == ["f0", "f1", "f2"][: terminal.data["lastEventId"]]


async def test_a_cancelled_task_closes_its_async_publisher_with_fail_at_once(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = await asyncio.to_thread(subscribe_in_background, ticket.url)
    started = asyncio.Event()

    async def agent() -> None:
        async with iq_client.apublisher(ticket.stream_id) as pub:
            pub.send("partial")
            started.set()
            await asyncio.sleep(30)

    task = asyncio.ensure_future(agent())
    await started.wait()
    await asyncio.sleep(0.2)
    cancelled = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    received = await subscription
    assert time.monotonic() - cancelled < 1.5, "iq:fail at once, not publisher_idle two minutes later"
    assert received.data == ["partial"]
    assert received.terminal == Terminal("iq:fail", {"reason": "publisher", "lastEventId": 1})


async def test_an_exception_in_the_sync_context_manager_reaches_iq_fail_at_once(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = await asyncio.to_thread(subscribe_in_background, ticket.url)

    def agent() -> None:
        with iq_client.publisher(ticket.stream_id) as pub:
            pub.send("partial")
            raise RuntimeError("the model failed")

    started = time.monotonic()
    with pytest.raises(RuntimeError):
        await asyncio.to_thread(agent)
    received = await subscription
    assert time.monotonic() - started < 1.5
    assert received.data == ["partial"]
    assert received.terminal == Terminal("iq:fail", {"reason": "publisher", "lastEventId": 1})


async def test_a_sync_publisher_created_used_and_closed_inside_a_running_loop(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    ticket = iq_client.ticket()
    subscription = await asyncio.to_thread(subscribe_in_background, ticket.url)
    with iq_client.publisher(ticket.stream_id) as pub:
        for i in range(3):
            assert pub.send(f"f{i}") is True
            await asyncio.sleep(0.05)
    assert pub.alive is False
    received = await subscription
    assert received.data == ["f0", "f1", "f2"]
    assert received.terminal == Terminal("iq:end", {"lastEventId": 3})

    other = iq_client.ticket()
    second = await asyncio.to_thread(subscribe_in_background, other.url)
    pub = iq_client.publisher(other.stream_id)
    pub.send("closed from a worker thread")
    assert (await asyncio.to_thread(pub.close)).last_event_id == 1
    assert (await second).data == ["closed from a worker thread"]


async def test_a_tickets_url_is_what_a_browser_subscribes_with(iq_servers: TestServers) -> None:
    iq = IQClient(
        "http://127.0.0.1:1",  # never reached: a ticket is minted locally
        iq_servers.app_key,
        iq_servers.app_secret,
        public_url=iq_servers.default.url,
        min_open=0,
    )
    ticket = iq.ticket()
    assert ticket.to_dict() == {
        "streamId": ticket.stream_id,
        "token": ticket.token,
        "url": f"{iq_servers.default.url}/{ticket.stream_id}?token={ticket.token}",
        "expiresAt": ticket.expires_at,
    }
    assert 3590 <= ticket.expires_at - time.time() <= 3600
    # The page subscribes first, then asks for the run with the stream id and the token, which the backend checks.
    subscription = await asyncio.to_thread(subscribe_in_background, ticket.url)
    assert iq.verify("sub", ticket.token, ticket.stream_id) is True
    assert iq.verify("sub", ticket.token, iq.new_stream_id()) is False
    publishing = IQClient(iq_servers.default.url, iq_servers.app_key, iq_servers.app_secret, min_open=0)
    async with publishing.apublisher(ticket.stream_id) as pub:
        pub.send('{"type":"text-delta","delta":"sunny"}')
    received = await subscription
    assert received.data == ['{"type":"text-delta","delta":"sunny"}']
    assert received.terminal == Terminal("iq:end", {"lastEventId": 1})
    refused = await asyncio.to_thread(subscribe, ticket.url.replace(ticket.stream_id, iq.new_stream_id()))
    assert (refused.status, refused.terminal) == (403, None), "a ticket opens its own stream and no other"
