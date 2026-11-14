"""A scripted transport: the test plays the server, one answer at a time, and reads what the writer wrote. One
fake serves both publishers: its sync side runs in the writer's thread, its async side on the loop."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from iq_stream import IQClient
from iq_stream._answer import NO_ANSWER, Answer
from iq_stream._transport import Watch

SECRET = "test-app-secret-0123456789abcdef0123"
STREAM = "Q2hhdC1zdHJlYW0tMDAwMQ"


async def until(check: Callable[[], Any], what: str = "the condition", seconds: float = 3.0) -> Any:
    deadline = time.monotonic() + seconds
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


def answer(status: int, **body: object) -> Answer:
    return Answer(status, body)


def open_(last_event_id: int) -> Answer:
    return answer(200, lastEventId=last_event_id, closed=False)


def closed_fail(last_event_id: int) -> Answer:
    return answer(200, lastEventId=last_event_id, closed=True, outcome="fail", reason="publisher")


def closed_end(last_event_id: int) -> Answer:
    return answer(200, lastEventId=last_event_id, closed=True, outcome="end")


CLOSED = closed_end(0)


class Pending:
    """An answer the test gives later."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.answer = NO_ANSWER

    def resolve(self, answer: Answer) -> None:
        self.answer = answer
        self._event.set()

    @property
    def resolved(self) -> bool:
        return self._event.is_set()


class FakeRequest:
    def __init__(self, stream_id: str, token: str, watch: Watch) -> None:
        self.stream_id = stream_id
        self.token = token
        self.watch = watch
        #: Every line written, without its newline; "" is a keepalive.
        self.lines: list[str] = []
        self.ended = False
        self.destroyed = False
        #: When true, writes do not complete: a server that stopped reading.
        self.stalled = False
        self._answer: Answer | None = None
        watch.on_destroy(self._on_destroy)

    def respond(self, status: int, **body: object) -> None:
        """The server answers; status 0 is a connection lost without an answer."""
        if self._answer is None:
            self._answer = Answer(status, body)
            self.watch.settle()

    @property
    def answered(self) -> bool:
        return self._answer is not None

    def _on_destroy(self) -> None:
        self.destroyed = True
        self.respond(0)

    def record(self, chunk: bytes) -> None:
        self.lines.extend(chunk.decode().split("\n")[:-1])

    @property
    def objects(self) -> list[dict[str, str]]:
        return [json.loads(line) for line in self.lines if line != ""]

    @property
    def data(self) -> list[str]:
        return [line["data"] for line in self.objects if "data" in line]

    @property
    def closes(self) -> list[str]:
        return [line["close"] for line in self.objects if "close" in line]

    @property
    def keepalives(self) -> int:
        return sum(1 for line in self.lines if line == "")


class FakeTransport:
    def __init__(self) -> None:
        self.requests: list[FakeRequest] = []
        #: The bodies of the short requests: "" is a probe.
        self.posts: list[str] = []
        self.post_tokens: list[str] = []
        #: The timeout each short request was given, in seconds.
        self.timeouts: list[float] = []
        #: Called with each request as it opens, to answer it at once.
        self.on_open: Callable[[FakeRequest, int], None] | None = None
        #: Answers each short request. By default the stream is empty and open, and a close line is accepted.
        self.on_post: Callable[[str, int], Answer | Pending] = lambda body, n: (
            closed_fail(0) if '"close"' in body else open_(0)
        )
        self.sync = _SyncSide(self)
        self.asyncio = _AsyncSide(self)

    def opened(self, stream_id: str, token: str, watch: Watch) -> FakeRequest:
        request = FakeRequest(stream_id, token, watch)
        self.requests.append(request)
        if self.on_open is not None:
            self.on_open(request, len(self.requests) - 1)
        return request

    def posted(self, token: str, body: bytes, timeout: float) -> Answer | Pending:
        text = body.decode()
        self.posts.append(text)
        self.post_tokens.append(token)
        self.timeouts.append(timeout)
        return self.on_post(text, len(self.posts) - 1)

    async def request(self, n: int) -> FakeRequest:
        """The nth held request, once the writer has opened it."""
        await until(lambda: len(self.requests) > n, f"request {n}")
        return self.requests[n]

    @property
    def probes(self) -> int:
        return sum(1 for body in self.posts if body == "")

    @property
    def closes(self) -> list[str]:
        """The close lines sent out of band."""
        return [json.loads(body)["close"] for body in self.posts if body != ""]


class _SyncSide:
    def __init__(self, fake: FakeTransport) -> None:
        self._fake = fake

    def stream(self, stream_id: str, token: str, body: Iterator[bytes], watch: Watch) -> Answer:
        request = self._fake.opened(stream_id, token, watch)
        for chunk in body:
            request.record(chunk)
            while request.stalled and not request.answered:
                time.sleep(0.002)
        request.ended = not watch.answered()
        while request._answer is None:  # pyright: ignore[reportPrivateUsage]
            time.sleep(0.002)
        return request._answer  # pyright: ignore[reportPrivateUsage]

    def post(self, stream_id: str, token: str, body: bytes, timeout: float, watch: Watch) -> Answer:
        reply = self._fake.posted(token, body, timeout)
        if isinstance(reply, Pending):
            while not reply.resolved and not watch.answered():
                time.sleep(0.002)
            return reply.answer
        return reply


async def _poll(check: Callable[[], bool]) -> None:
    # State shared with the writer's thread in the sync tests: polled, since no asyncio.Event can be set from there.
    while not check():  # noqa: ASYNC110
        await asyncio.sleep(0.002)


class _AsyncSide:
    def __init__(self, fake: FakeTransport) -> None:
        self._fake = fake

    async def stream(self, stream_id: str, token: str, body: AsyncIterator[bytes], watch: Watch) -> Answer:
        request = self._fake.opened(stream_id, token, watch)
        async for chunk in body:
            request.record(chunk)
            await _poll(lambda: not request.stalled or request.answered)
        request.ended = not watch.answered()
        await _poll(lambda: request.answered)
        assert request._answer is not None  # pyright: ignore[reportPrivateUsage]
        return request._answer  # pyright: ignore[reportPrivateUsage]

    async def post(self, stream_id: str, token: str, body: bytes, timeout: float, watch: Watch) -> Answer:
        reply = self._fake.posted(token, body, timeout)
        if isinstance(reply, Pending):
            await _poll(lambda: reply.resolved or watch.answered())
            return reply.answer
        return reply

    async def aclose(self) -> None:
        pass


class Harness:
    """A client over a fake transport, for one kind of publisher, with the errors its publishers reported."""

    def __init__(self, kind: str, **options: Any) -> None:
        self.kind = kind
        self.transport = FakeTransport()
        self.errors: list[Any] = []
        settings: dict[str, Any] = {"min_open": 0.0, "error_handler": self.errors.append, **options}
        self.client = IQClient("http://iq.test", "demo-app", SECRET, **settings)
        core = self.client._core  # pyright: ignore[reportPrivateUsage]
        core.sync_transport = lambda: self.transport.sync
        core.async_transport = lambda: self.transport.asyncio

    def publisher(self) -> Any:
        return self.client.publisher(STREAM) if self.kind == "sync" else self.client.apublisher(STREAM)

    def close(self, publisher: Any, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """``close()`` of either publisher as an awaitable: the sync one runs in a thread, as async code calls it."""
        if self.kind == "sync":
            return asyncio.to_thread(publisher.close, *args, **kwargs)
        return publisher.close(*args, **kwargs)

    async def request(self, n: int) -> FakeRequest:
        return await self.transport.request(n)
