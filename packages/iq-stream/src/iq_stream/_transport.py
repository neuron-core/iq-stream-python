"""What the writer needs from HTTP, over httpx.

httpx sends the whole body of a request before it reads the answer. Two things make up for it. httpcore's
``trace`` extension hands over the connection's socket as it opens: the writer then looks at it before every
write and while it waits, and stops the body as soon as the server has answered or dropped the connection (on
plain connections; TLS records such as session tickets make a TLS socket readable without an answer). And
``destroy()`` shuts that socket down from any thread, which ends a write the server stopped reading.
"""

from __future__ import annotations

import contextlib
import os
import select
import socket
import ssl
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol

import httpx

from ._answer import NO_ANSWER, Answer, parse_body

#: How long a connection may take to open.
CONNECT_TIMEOUT = 5.0
#: How long the answer to a held request may take once its body has ended.
ANSWER_TIMEOUT = 10.0


def _version() -> str:
    try:
        return version("iq-stream")
    except PackageNotFoundError:  # pragma: no cover - a source tree that is not installed
        return "0"


USER_AGENT = f"iq-stream-python/{_version()}"
# Each request has a connection of its own: a stream's request is held for as long as the answer takes, and a
# pooled connection the server has closed would cost a failed request.
_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=0)


def _readable(sock: socket.socket) -> bool:
    try:
        if hasattr(select, "poll"):
            poller = select.poll()
            poller.register(sock, select.POLLIN | select.POLLPRI | select.POLLERR | select.POLLHUP)
            return bool(poller.poll(0))
        ready, _, _ = select.select([sock], [], [], 0)  # pragma: no cover - Windows
        return bool(ready)  # pragma: no cover
    except (OSError, ValueError):
        # A socket that can no longer be polled is gone.
        return True


class Watch:
    """One request's connection, as far as the writer needs it: whether the server has answered, and a way to
    drop it."""

    def __init__(self, wake: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._wake = wake
        self._sock: socket.socket | None = None
        self._tls = False
        self._settled = False
        self._destroyed = False
        self._on_destroy: list[Callable[[], None]] = []

    def trace(self, event: str, info: Mapping[str, object]) -> None:
        """httpcore's ``trace`` extension: takes the socket of each connection step that completes."""
        if not event.endswith(".complete"):
            return
        get_extra_info = getattr(info.get("return_value"), "get_extra_info", None)
        if not callable(get_extra_info):
            return
        sock = get_extra_info("socket")
        if not isinstance(sock, socket.socket) and not callable(getattr(sock, "shutdown", None)):
            return
        with self._lock:
            self._sock = sock  # pyright: ignore[reportAttributeAccessIssue]
            self._tls = self._tls or get_extra_info("ssl_object") is not None
            destroyed = self._destroyed
        if destroyed:
            self._shutdown()

    async def atrace(self, event: str, info: Mapping[str, object]) -> None:
        self.trace(event, info)

    def answered(self) -> bool:
        """True once the server has answered or the connection is lost: nothing more may be written."""
        if self._settled or self._destroyed:
            return True
        sock = self._sock
        return sock is not None and not self._tls and _readable(sock)

    def settle(self) -> None:
        """Says that the server has answered (for transports that know it without a socket)."""
        self._settled = True
        if self._wake is not None:
            self._wake()

    def on_destroy(self, callback: Callable[[], None]) -> None:
        self._on_destroy.append(callback)

    def destroy(self) -> None:
        """Drops the connection, from any thread. A write or a read in progress on it ends at once."""
        with self._lock:
            if self._destroyed:
                return
            self._destroyed = True
        self._shutdown()
        for callback in self._on_destroy:
            callback()
        if self._wake is not None:
            self._wake()

    def _shutdown(self) -> None:
        sock = self._sock
        if sock is None:
            return
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)


class SyncTransport(Protocol):
    def stream(self, stream_id: str, token: str, body: Iterator[bytes], watch: Watch) -> Answer:
        """A chunked ``POST`` whose body is pulled from ``body`` as it is written. Never raises for the network."""
        ...

    def post(self, stream_id: str, token: str, body: bytes, timeout: float, watch: Watch) -> Answer:
        """A short ``POST`` with its whole body: a probe, a close line, a reservation."""
        ...


class AsyncTransport(Protocol):
    async def stream(self, stream_id: str, token: str, body: AsyncIterator[bytes], watch: Watch) -> Answer: ...

    async def post(self, stream_id: str, token: str, body: bytes, timeout: float, watch: Watch) -> Answer: ...

    async def aclose(self) -> None: ...


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/x-ndjson", "User-Agent": USER_AGENT}


_STREAM_TIMEOUT = httpx.Timeout(connect=CONNECT_TIMEOUT, read=ANSWER_TIMEOUT, write=None, pool=None)


class Connector:
    """How this backend reaches the server: the address, and httpx clients configured for it, made on demand.
    Nothing is opened before the first request."""

    def __init__(self, base: str, uds: str | None, trust_env: bool, verify: bool | ssl.SSLContext) -> None:
        self.base = base
        self.uds = uds
        self.trust_env = trust_env
        self._verify = verify
        self._lock = threading.Lock()
        self._context: ssl.SSLContext | None = None
        self._sync: SyncHTTP | None = None

    def url(self, stream_id: str) -> str:
        return f"{self.base}/{stream_id}"

    def _ssl(self) -> ssl.SSLContext:
        if isinstance(self._verify, ssl.SSLContext):
            return self._verify
        with self._lock:
            # One context for every client of this IQClient: loading the CA bundle costs milliseconds.
            if self._context is None:
                self._context = httpx.create_ssl_context(verify=self._verify, trust_env=self.trust_env)
            return self._context

    def sync_client(self) -> httpx.Client:
        if self.uds is not None:
            return httpx.Client(transport=httpx.HTTPTransport(uds=self.uds, limits=_LIMITS), trust_env=False)
        return httpx.Client(limits=_LIMITS, verify=self._ssl(), trust_env=self.trust_env)

    def async_client(self) -> httpx.AsyncClient:
        if self.uds is not None:
            return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=self.uds, limits=_LIMITS), trust_env=False)
        return httpx.AsyncClient(limits=_LIMITS, verify=self._ssl(), trust_env=self.trust_env)

    def sync(self) -> SyncHTTP:
        """The transport the synchronous publishers of this client share. httpx's Client is thread-safe."""
        with self._lock:
            if self._sync is None:
                self._sync = SyncHTTP(self)
            return self._sync


class SyncHTTP:
    def __init__(self, connector: Connector) -> None:
        self._connector = connector
        self._lock = threading.Lock()
        self._client: httpx.Client | None = None
        self._pid = 0

    def _http(self) -> httpx.Client:
        with self._lock:
            # A forked worker makes a client of its own: the parent's may hold locks taken at the fork.
            if self._client is None or self._pid != os.getpid():
                self._client = self._connector.sync_client()
                self._pid = os.getpid()
            return self._client

    def stream(self, stream_id: str, token: str, body: Iterator[bytes], watch: Watch) -> Answer:
        try:
            response = self._http().post(
                self._connector.url(stream_id),
                content=body,
                headers=_headers(token),
                timeout=_STREAM_TIMEOUT,
                extensions={"trace": watch.trace},
            )
        except httpx.HTTPError:
            return NO_ANSWER
        return Answer(response.status_code, parse_body(response.content))

    def post(self, stream_id: str, token: str, body: bytes, timeout: float, watch: Watch) -> Answer:
        try:
            response = self._http().post(
                self._connector.url(stream_id),
                content=body,
                headers=_headers(token),
                timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT)),
                extensions={"trace": watch.trace},
            )
        except httpx.HTTPError:
            return NO_ANSWER
        return Answer(response.status_code, parse_body(response.content))


class AsyncHTTP:
    """An httpx AsyncClient of its own, made on the loop that uses it and closed with ``aclose()``."""

    def __init__(self, connector: Connector) -> None:
        self._connector = connector
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._connector.async_client()
        return self._client

    async def stream(self, stream_id: str, token: str, body: AsyncIterator[bytes], watch: Watch) -> Answer:
        try:
            response = await self._http().post(
                self._connector.url(stream_id),
                content=body,
                headers=_headers(token),
                timeout=_STREAM_TIMEOUT,
                extensions={"trace": watch.atrace},
            )
        except httpx.HTTPError:
            return NO_ANSWER
        return Answer(response.status_code, parse_body(response.content))

    async def post(self, stream_id: str, token: str, body: bytes, timeout: float, watch: Watch) -> Answer:
        try:
            response = await self._http().post(
                self._connector.url(stream_id),
                content=body,
                headers=_headers(token),
                timeout=httpx.Timeout(timeout, connect=min(timeout, CONNECT_TIMEOUT)),
                extensions={"trace": watch.atrace},
            )
        except httpx.HTTPError:
            return NO_ANSWER
        return Answer(response.status_code, parse_body(response.content))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
