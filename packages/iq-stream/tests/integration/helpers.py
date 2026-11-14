from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from iq_stream import IQClient
from iq_stream.testing import BackgroundSubscription, ServerAddress, TestServers, subscribe_in_background


async def until(check: Callable[[], Any], what: str = "the condition", seconds: float = 5.0) -> Any:
    deadline = time.monotonic() + seconds
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


class Proxy:
    """A TCP proxy in front of a server, listening on a local port or on a Unix socket path. ``kill()`` drops
    every connection through it, which is how a test loses a publisher's connection from the client side without
    touching the server."""

    def __init__(self, target: str, path: str | None = None) -> None:
        parts = urlsplit(target)
        self._upstream = (parts.hostname or "127.0.0.1", parts.port or 80)
        self._lock = threading.Lock()
        self._pairs: set[tuple[socket.socket, socket.socket]] = set()
        self._sockets: list[socket.socket] = []
        self._threads: list[threading.Thread] = []
        self.connections = 0
        if path is None:
            self._listener = socket.create_server(("127.0.0.1", 0))
            self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        else:
            self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener.bind(path)
            self._listener.listen()
            self.url = f"http+unix://{path}"
        self._accepting = threading.Thread(target=self._accept, daemon=True)
        self._accepting.start()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(self._upstream)
            pair = (client, upstream)
            with self._lock:
                self.connections += 1
                self._pairs.add(pair)
                self._sockets.extend(pair)
            for source, target in ((client, upstream), (upstream, client)):
                pump = threading.Thread(target=self._pump, args=(source, target, pair), daemon=True)
                self._threads.append(pump)
                pump.start()

    def _pump(self, source: socket.socket, target: socket.socket, pair: tuple[socket.socket, socket.socket]) -> None:
        try:
            while data := source.recv(65536):
                target.sendall(data)
        except OSError:
            pass
        self._drop(pair)

    def _drop(self, pair: tuple[socket.socket, socket.socket]) -> None:
        with self._lock:
            self._pairs.discard(pair)
        for sock in pair:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    def kill(self) -> None:
        with self._lock:
            pairs = list(self._pairs)
        for pair in pairs:
            self._drop(pair)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._listener.shutdown(socket.SHUT_RDWR)
        self._listener.close()
        self._accepting.join(2)
        self.kill()
        for thread in self._threads:
            thread.join(2)
        for sock in self._sockets:
            sock.close()


class Run:
    """What a scenario needs around a client, for one kind of publisher: its subscriber, its publisher, its close,
    and the errors its publishers reported."""

    def __init__(self, kind: str, servers: TestServers) -> None:
        self.kind = kind
        self.servers = servers
        self.errors: list[Any] = []

    def client(self, server: ServerAddress | str, **options: Any) -> IQClient:
        settings: dict[str, Any] = {
            "app_key": self.servers.app_key,
            "app_secret": self.servers.app_secret,
            "min_open": 0.0,
            "error_handler": self.errors.append,
            **options,
        }
        return IQClient(server if isinstance(server, str) else server.url, **settings)

    def publisher(self, client: IQClient, stream_id: str) -> Any:
        return client.publisher(stream_id) if self.kind == "sync" else client.apublisher(stream_id)

    async def close(self, publisher: Any, outcome: str = "end", **options: Any) -> Any:
        if self.kind == "sync":
            return await asyncio.to_thread(publisher.close, outcome, **options)
        return await publisher.close(outcome, **options)

    async def listen(
        self, client: IQClient, server: ServerAddress, stream_id: str | None = None
    ) -> tuple[str, BackgroundSubscription]:
        """Subscribes to a new stream on ``server``, straight to the server whatever the client's url is, and
        returns once the server has accepted the subscription."""
        stream_id = stream_id or client.new_stream_id()
        token = client.token("sub", stream_id)
        subscription = await asyncio.to_thread(subscribe_in_background, server.url, stream_id, token)
        return stream_id, subscription
