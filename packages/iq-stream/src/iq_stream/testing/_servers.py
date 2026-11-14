from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, TracebackType

import httpx

#: The settings of the "short" server: the timeouts and limits of a stream, reachable in seconds.
SHORT_SETTINGS: Mapping[str, str] = MappingProxyType(
    {
        "PUBLISH_READ_TIMEOUT_MS": "2000",
        "STREAM_IDLE_TIMEOUT_MS": "6000",
        "PENDING_TIMEOUT_MS": "5000",
        "RETRY_MS": "500",
        "RETENTION_MS": "8000",
        "MAX_FRAME_BYTES": "1024",
        "MAX_STREAM_BYTES": "8192",
    }
)

DEFAULT_IMAGE = "ghcr.io/neuron-core/iq-stream:1"
DEFAULT_APP_KEY = "demo-app"
DEFAULT_APP_SECRET = "test-app-secret-0123456789abcdef0123"


def _docker(*args: str) -> str:
    try:
        completed = subprocess.run(["docker", *args], capture_output=True, text=True, check=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError("docker is not installed, or not on PATH") from None
    except subprocess.CalledProcessError as error:
        # Docker's own words, and not the command line, which carries the app secret.
        detail = (error.stderr or "").strip() or "is Docker running?"
        raise RuntimeError(f"docker {args[0]} failed: {detail}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"docker {args[0]} did not finish in 120 s") from None
    return completed.stdout.strip()


def healthy(url: str) -> bool:
    """Whether an IQ-Stream server answers ``/healthz`` with 200 on ``url``."""
    try:
        return httpx.get(f"{url.rstrip('/')}/healthz", timeout=1.0, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False


def _wait_healthy(url: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if healthy(url):
            return True
        time.sleep(0.1)
    return False


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class StartedServer:
    """An IQ-Stream server in a container of its own, for a test that kills or restarts it."""

    def __init__(self, url: str, container: str, image: str) -> None:
        self.url = url
        #: The container's id.
        self.container = container
        self.image = image

    def kill(self) -> None:
        """Kills the server without letting it drain, as a crash does."""
        _docker("kill", self.container)

    def start(self) -> None:
        """Starts it again, empty, on the same address, and waits until it answers."""
        _docker("start", self.container)
        if not _wait_healthy(self.url, 20):
            raise RuntimeError(f"The IQ-Stream server from {self.image} did not answer again on {self.url}/healthz.")

    def stop(self) -> None:
        """Removes the container."""
        _docker("rm", "-f", self.container)

    def __enter__(self) -> StartedServer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.stop()


def start_server(
    *,
    env: Mapping[str, str] | None = None,
    port: int | None = None,
    image: str | None = None,
    app_key: str | None = None,
    app_secret: str | None = None,
) -> StartedServer:
    """Starts a server in Docker on ``127.0.0.1``, on a free port unless one is given, and waits until it answers.

    The image is ``image``, else ``IQ_STREAM_TEST_IMAGE``, else ``ghcr.io/neuron-core/iq-stream:1``.
    """
    image = image or os.environ.get("IQ_STREAM_TEST_IMAGE") or DEFAULT_IMAGE
    port = port or free_port()
    variables = {"APP_KEY": app_key or DEFAULT_APP_KEY, "APP_SECRET": app_secret or DEFAULT_APP_SECRET, **(env or {})}
    flags = [part for name, value in variables.items() for part in ("-e", f"{name}={value}")]
    container = _docker("run", "-d", "-p", f"127.0.0.1:{port}:8080", *flags, image)
    server = StartedServer(f"http://127.0.0.1:{port}", container, image)
    if not _wait_healthy(server.url, 20):
        server.stop()
        raise RuntimeError(f"The IQ-Stream server from {image} did not answer on {server.url}/healthz.")
    return server


@dataclass(frozen=True)
class ServerAddress:
    url: str


@dataclass
class TestServers:
    """The two servers the integration tests publish to."""

    __test__ = False

    #: A server with default settings.
    default: ServerAddress
    #: A server with ``SHORT_SETTINGS``.
    short: ServerAddress
    app_key: str
    app_secret: str
    started: list[StartedServer] = field(default_factory=lambda: [])

    def stop(self) -> None:
        """Removes the containers this call started. Servers that were already running are left alone."""
        while self.started:
            self.started.pop().stop()


def test_servers() -> TestServers:
    """Finds the two servers: each is the one its variable names (``IQ_STREAM_TEST_URL``,
    ``IQ_STREAM_TEST_SHORT_URL``), else the one answering on its default address (``http://127.0.0.1:8080``,
    ``http://127.0.0.1:8081``), else a container started from ``IQ_STREAM_TEST_IMAGE``. The credentials are
    ``IQ_STREAM_TEST_APP_KEY`` and ``IQ_STREAM_TEST_APP_SECRET``, else the defaults of the servers' test setup."""
    app_key = os.environ.get("IQ_STREAM_TEST_APP_KEY") or DEFAULT_APP_KEY
    app_secret = os.environ.get("IQ_STREAM_TEST_APP_SECRET") or DEFAULT_APP_SECRET
    started: list[StartedServer] = []

    def find(variable: str, fallback: str, env: Mapping[str, str] | None) -> ServerAddress:
        named = os.environ.get(variable)
        if named:
            return ServerAddress(named.rstrip("/"))
        if healthy(fallback):
            return ServerAddress(fallback)
        server = start_server(env=env, app_key=app_key, app_secret=app_secret)
        started.append(server)
        return ServerAddress(server.url)

    try:
        return TestServers(
            default=find("IQ_STREAM_TEST_URL", "http://127.0.0.1:8080", None),
            short=find("IQ_STREAM_TEST_SHORT_URL", "http://127.0.0.1:8081", SHORT_SETTINGS),
            app_key=app_key,
            app_secret=app_secret,
            started=started,
        )
    except BaseException:
        for server in started:
            server.stop()
        raise


test_servers.__test__ = False  # pyright: ignore[reportFunctionMemberAccess]
