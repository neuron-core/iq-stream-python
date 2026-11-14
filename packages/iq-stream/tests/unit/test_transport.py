"""The socket watch the writer relies on to see an answer without writing, and to drop a connection."""

from __future__ import annotations

import socket
import threading
import time

import httpx

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
        watch.trace("connection.connect_unix_socket.complete", {"return_value": _Stream(client)})
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
    connector = Connector("http://127.0.0.1:1", None, False, True)
    assert connector.url("abc") == "http://127.0.0.1:1/abc"
    assert connector.sync() is connector.sync(), "one transport shared by the synchronous publishers"
    with connector.sync_client() as client:
        assert client.trust_env is False
    unix = Connector("http://localhost", "/tmp/iq.sock", False, True)
    with unix.sync_client() as client:
        assert isinstance(client, httpx.Client)
    assert USER_AGENT.startswith("iq-stream-python/")
