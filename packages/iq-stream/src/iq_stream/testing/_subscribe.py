from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Generator
from dataclasses import dataclass, field
from time import monotonic
from typing import cast
from urllib.parse import quote

import httpx

from .._protocol import TERMINAL_EVENTS
from .._transport import Watch


@dataclass(frozen=True)
class SubscribedFrame:
    """A data event as a subscriber received it."""

    id: str
    #: ``message`` for an unnamed frame.
    event: str
    data: str
    #: Seconds from the start of the subscription to the frame's arrival.
    at: float


@dataclass(frozen=True)
class Terminal:
    """The server's terminal event: ``iq:end``, ``iq:fail`` or ``iq:gone``, with its parsed data."""

    event: str
    data: dict[str, object]


@dataclass
class Subscription:
    """Everything a subscription delivered."""

    #: The data events, in the order they arrived.
    frames: list[SubscribedFrame] = field(default_factory=lambda: [])
    #: The terminal event, or ``None`` when the response ended without one.
    terminal: Terminal | None = None
    #: The HTTP status; 0 when the server could not be reached.
    status: int = 0

    @property
    def data(self) -> list[str]:
        """The data of every frame, in order."""
        return [frame.data for frame in self.frames]


class _Parser:
    """The SSE fields of a stream, as an EventSource reads them."""

    def __init__(self, result: Subscription, started: float) -> None:
        self._result = result
        self._started = started
        self._id = ""
        self._event = ""
        self._data: list[str] = []

    def line(self, text: str) -> bool:
        """Takes one line; True once the terminal event has arrived."""
        if text == "":
            self._dispatch()
        elif not text.startswith(":"):
            name, colon, value = text.partition(":")
            value = value[1:] if colon and value.startswith(" ") else value
            if name == "id":
                self._id = value
            elif name == "event":
                self._event = value
            elif name == "data":
                self._data.append(value)
        return self._result.terminal is not None

    def _dispatch(self) -> None:
        if self._data or self._event:
            text = "\n".join(self._data)
            if self._event in TERMINAL_EVENTS:
                try:
                    parsed: object = json.loads(text)
                except ValueError:
                    parsed = None
                data = cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}
                self._result.terminal = Terminal(self._event, data)
            elif not self._event.startswith("iq:"):
                at = round(monotonic() - self._started, 3)
                self._result.frames.append(SubscribedFrame(self._id, self._event or "message", text, at))
        self._event = ""
        self._data = []


def _target(url: str, stream_id: str | None, token: str | None) -> str:
    if stream_id is None and token is None:
        return url
    if stream_id is None or token is None:
        raise ValueError("Give both stream_id and token, or neither and a complete subscribe URL.")
    return f"{url.rstrip('/')}/{stream_id}?token={quote(token, safe='')}"


def _headers(last_event_id: str | int | None) -> dict[str, str]:
    headers = {"Accept": "text/event-stream"}
    if last_event_id is not None:
        headers["Last-Event-ID"] = str(last_event_id)
    return headers


def _read(
    target: str, last_event_id: str | int | None, timeout: float, watch: Watch, opened: threading.Event | None
) -> Subscription:
    started = monotonic()
    result = Subscription()
    parser = _Parser(result, started)
    # The whole subscription is bounded: the connection is dropped when the time is up.
    timer = threading.Timer(timeout, watch.destroy)
    timer.daemon = True
    timer.start()
    try:
        with (
            httpx.Client(trust_env=False, timeout=httpx.Timeout(timeout, connect=5.0)) as client,
            client.stream(
                "GET", target, headers=_headers(last_event_id), extensions={"trace": watch.trace}
            ) as response,
        ):
            result.status = response.status_code
            if response.status_code == 200 and opened is not None:
                opened.set()
            for line in response.iter_lines():
                if parser.line(line):
                    break
    except httpx.HTTPError:
        pass
    finally:
        timer.cancel()
    return result


def subscribe(
    url: str,
    stream_id: str | None = None,
    token: str | None = None,
    *,
    last_event_id: str | int | None = None,
    timeout: float = 30.0,
) -> Subscription:
    """Subscribes to a stream as a browser would and returns everything it delivered, once the terminal event
    arrives, the response ends, or ``timeout`` seconds have passed.

    ``url`` is the server's address and the stream is ``stream_id`` with the ``sub`` token ``token``; with only
    ``url``, it is a complete subscribe URL, such as a ticket's.
    """
    return _read(_target(url, stream_id, token), last_event_id, timeout, Watch(), None)


async def asubscribe(
    url: str,
    stream_id: str | None = None,
    token: str | None = None,
    *,
    last_event_id: str | int | None = None,
    timeout: float = 30.0,
) -> Subscription:
    """``subscribe()`` for asyncio."""
    target = _target(url, stream_id, token)
    result = Subscription()
    parser = _Parser(result, monotonic())

    async def read() -> None:
        async with (
            httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(timeout, connect=5.0)) as client,
            client.stream("GET", target, headers=_headers(last_event_id)) as response,
        ):
            result.status = response.status_code
            async for line in response.aiter_lines():
                if parser.line(line):
                    break

    with contextlib.suppress(httpx.HTTPError, asyncio.TimeoutError):
        await asyncio.wait_for(read(), timeout)
    return result


class BackgroundSubscription:
    """A subscription that runs in a thread of its own: started before the publisher, collected after it. It
    works the same from synchronous tests and from asyncio ones (``await subscription``)."""

    def __init__(self, target: str, last_event_id: str | int | None, timeout: float) -> None:
        self._opened = threading.Event()
        self._done = threading.Event()
        self._watch = Watch()
        self._result: Subscription | None = None

        def run() -> None:
            try:
                self._result = _read(target, last_event_id, timeout, self._watch, self._opened)
            finally:
                self._done.set()

        self._thread = threading.Thread(target=run, name="iq-stream subscriber", daemon=True)
        self._thread.start()

    def wait_open(self, timeout: float = 5.0) -> bool:
        """Waits until the server has accepted the subscription; False if it refused it or did not answer."""
        deadline = monotonic() + timeout
        while not self._opened.wait(0.01):
            if self._done.is_set() or monotonic() > deadline:
                return self._opened.is_set()
        return True

    def result(self, timeout: float | None = None) -> Subscription:
        """Waits for the subscription to end and returns what it delivered."""
        if not self._done.wait(timeout):
            raise TimeoutError("The subscription did not end in time.")
        assert self._result is not None
        return self._result

    async def aresult(self, timeout: float | None = None) -> Subscription:
        return await asyncio.to_thread(self.result, timeout)

    def __await__(self) -> Generator[object, None, Subscription]:
        return self.aresult().__await__()

    def close(self) -> None:
        """Ends the subscription now."""
        self._watch.destroy()


def subscribe_in_background(
    url: str,
    stream_id: str | None = None,
    token: str | None = None,
    *,
    last_event_id: str | int | None = None,
    timeout: float = 30.0,
    wait_open: float | None = 5.0,
) -> BackgroundSubscription:
    """Starts ``subscribe()`` in a thread and returns at once, by default after the server has accepted the
    subscription (``wait_open`` seconds at most; ``None`` does not wait)."""
    subscription = BackgroundSubscription(_target(url, stream_id, token), last_event_id, timeout)
    if wait_open is not None:
        subscription.wait_open(wait_open)
    return subscription
