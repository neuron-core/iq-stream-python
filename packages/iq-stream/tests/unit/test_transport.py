"""The socket watch the writer relies on to see an answer without writing, and to drop a connection."""

from __future__ import annotations

import asyncio
import socket
import ssl
import threading
import time
from typing import ClassVar

import pytest

from iq_stream import AsyncPublisher, IQClient, IQError, new_stream_id
from iq_stream._transport import USER_AGENT, Connector, Watch


class _Stream:
    def __init__(self, sock: socket.socket, tls: bool = False) -> None:
        self._sock = sock
        self._tls = tls

    def get_extra_info(self, name: str) -> object:
        return {"socket": self._sock, "ssl_object": object() if self._tls else None}.get(name)


def test_a_watch_sees_an_answer_on_its_socket_without_writing() -> None:
    client, server = socket.socketpair()
    with client, server:
        watch = Watch()
        assert watch.answered() is False, "no socket yet"
        watch.trace("connection.connect_tcp.started", {})
        watch.trace("connection.connect_tcp.complete", {"return_value": _Stream(client)})
        assert watch.answered() is False
        server.sendall(b"HTTP/1.1 408 Request Timeout\r\n")
        assert watch.answered() is True


def test_a_watch_sees_a_dropped_connection() -> None:
    client, server = socket.socketpair()
    with client:
        watch = Watch()
        watch.trace("connection.connect_tcp.complete", {"return_value": _Stream(client)})
        server.close()
        assert watch.answered() is True


def test_a_tls_socket_is_not_taken_as_answered_since_tls_records_arrive_unasked() -> None:
    client, server = socket.socketpair()
    with client, server:
        watch = Watch()
        watch.trace("connection.start_tls.complete", {"return_value": _Stream(client, tls=True)})
        server.sendall(b"a session ticket")
        assert watch.answered() is False


def test_destroy_ends_a_blocked_read_in_another_thread_and_calls_back() -> None:
    client, server = socket.socketpair()
    with client, server:
        woken: list[str] = []
        watch = Watch(lambda: woken.append("wake"))
        dropped: list[str] = []
        watch.on_destroy(lambda: dropped.append("dropped"))
        watch.trace("connection.connect_tcp.complete", {"return_value": _Stream(client)})
        received: list[bytes] = []
        reader = threading.Thread(target=lambda: received.append(client.recv(10)))
        reader.start()
        time.sleep(0.05)
        watch.destroy()
        reader.join(2)
        assert not reader.is_alive()
        assert received == [b""]
        assert (watch.answered(), dropped, woken) == (True, ["dropped"], ["wake"])
        watch.destroy()
        assert dropped == ["dropped"], "once"


def test_a_watch_destroyed_before_its_connection_opens_drops_it_as_it_opens() -> None:
    client, server = socket.socketpair()
    with client, server:
        watch = Watch()
        watch.destroy()
        watch.trace("connection.connect_tcp.complete", {"return_value": _Stream(client)})
        server.settimeout(1)
        assert server.recv(10) == b"", "the client side was shut down"


def test_the_connector_makes_clients_that_keep_no_connection() -> None:
    connector = Connector("http://127.0.0.1:1", False, True)
    assert connector.url("abc") == "http://127.0.0.1:1/abc"
    assert connector.sync() is connector.sync(), "one transport shared by the synchronous publishers"
    with connector.sync_client() as client:
        assert client.trust_env is False
    assert USER_AGENT.startswith("iq-stream-python/")


class _RefusesShutdown:
    """uvloop's ``PseudoSocket``: the transport's socket, whose ``shutdown()`` raises TypeError."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def fileno(self) -> int:
        return self._sock.fileno()

    def shutdown(self, how: int) -> None:
        raise TypeError("transport sockets do not support shutdown() method")


def test_a_watch_drops_a_connection_whose_socket_refuses_shutdown() -> None:
    client, server = socket.socketpair()
    with client, server:
        dropped: list[str] = []
        watch = Watch(lambda: dropped.append("wake"))
        watch.on_destroy(lambda: dropped.append("dropped"))
        watch.trace("connection.connect_tcp.complete", {"return_value": _Stream(_RefusesShutdown(client))})  # type: ignore[arg-type]
        watch.destroy()
        server.settimeout(1)
        assert server.recv(10) == b"", "the connection was shut down all the same"
        assert dropped == ["dropped", "wake"]


class _Unwrapped(socket.socket):
    """Sits under ``SSLSocket`` in the MRO: its ``shutdown()`` runs once ``SSLSocket.shutdown()`` has unwrapped the
    socket and before the connection is shut down, the moment a write on the writer's thread would go out in clear."""

    leaked: ClassVar[list[bool]] = []

    def shutdown(self, how: int) -> None:
        self.leaked.append(True)
        self.send(b"PLAINTEXT")
        super().shutdown(how)


class _Spy(ssl.SSLSocket, _Unwrapped):
    pass


def test_a_tls_socket_is_shut_down_without_being_unwrapped_first() -> None:
    context = ssl.create_default_context()
    context.sslsocket_class = _Spy
    client, server = socket.socketpair()
    with server, context.wrap_socket(client, server_hostname="localhost", do_handshake_on_connect=False) as tls:
        watch = Watch()
        watch.trace("connection.start_tls.complete", {"return_value": _Stream(tls, tls=True)})
        watch.destroy()
        server.settimeout(1)
        assert server.recv(100) == b"", "nothing in clear, and the connection is shut down"
        assert _Unwrapped.leaked == []


class _BlackHole:
    """A server that reads a request's head and nothing more, and never answers: a stalled connection. A second
    connection for the same stream (the out-of-band close of a publisher that gave up) is refused at once."""

    def __init__(self) -> None:
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        self.url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        self.held: dict[str, socket.socket] = {}
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = connection.recv(1)
                if not chunk:
                    break
                head += chunk
            path = head.split(b" ")[1].decode() if head.count(b" ") >= 2 else ""
            if path in self.held:
                connection.close()
            else:
                self.held[path] = connection

    def __enter__(self) -> _BlackHole:
        return self

    def __exit__(self, *exc: object) -> None:
        self._listener.close()
        for connection in self.held.values():
            connection.close()


def test_under_uvloop_abort_a_close_timeout_and_a_full_queue_drop_the_stalled_connection() -> None:
    uvloop = pytest.importorskip("uvloop")
    errors: list[IQError] = []

    async def stalled(iq: IQClient, server: _BlackHole) -> AsyncPublisher:
        pub = iq.apublisher(new_stream_id())
        pub.send("a")
        while f"/{pub.stream_id}" not in server.held:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        return pub

    async def main(server: _BlackHole) -> list[str]:
        iq = IQClient(server.url, "demo-app", "x" * 32, min_open=0, retry_budget=0, error_handler=errors.append)
        seen: list[str] = []

        aborted = await stalled(iq, server)
        aborted.abort()  # raised TypeError under uvloop
        await asyncio.wait_for(aborted._task, 3)  # pyright: ignore[reportPrivateUsage] - the request was dropped
        seen.append("abort")

        timed_out = await stalled(iq, server)
        outcome = await timed_out.close(timeout=0.3)
        await asyncio.wait_for(timed_out._task, 3)  # pyright: ignore[reportPrivateUsage]
        seen.append(str(outcome.error))

        full = await stalled(iq, server)
        frame = "x" * 60_000
        for _ in range(400):
            if not full.send(frame):  # raised TypeError under uvloop
                break
            await asyncio.sleep(0.002)
        await asyncio.wait_for(full._task, 3)  # pyright: ignore[reportPrivateUsage]
        seen.append(str((await full.close()).error))
        return seen

    with _BlackHole() as server:
        assert uvloop.run(main(server)) == ["abort", "close_timeout", "queue_full"]
    assert [error.code for error in errors] == ["close_timeout", "queue_full"], "the handler ran for each"
