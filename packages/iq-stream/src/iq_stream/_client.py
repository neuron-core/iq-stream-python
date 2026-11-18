from __future__ import annotations

import asyncio
import ipaddress
import math
import os
import re
import ssl
import time
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlsplit

from ._answer import Answer, PublishResult, refusal, to_result
from ._apublisher import AsyncPublisher
from ._core import PUB_TOKEN_TTL, Core
from ._errors import IQError
from ._protocol import (
    DEFAULT_MAX_FRAME_BYTES,
    DEFAULT_TOKEN_TTL,
    Outcome,
    Scope,
    Ticket,
    check_outcome,
    check_scope,
    check_seconds,
    check_stream_id,
)
from ._publisher import Publisher
from ._token import claims, new_stream_id, sign
from ._token import verify as verify_token
from ._transport import AsyncHTTP, AsyncTransport, Connector, Watch
from ._writer import POST_TIMEOUT, Command, Pause, Post, Settings, adrive, close_stream_within, drive

_APP_KEY = re.compile(r"[!-~]{1,256}")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _number(name: str, value: object, minimum: float, *, integer: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (integer and not isinstance(value, int))
    ):
        raise IQError(f"iq-stream: invalid option {name}.")
    return value


def _private_host(host: str) -> bool:
    """A loopback or private address, or a name that can only be one: a proxy between a backend and a server on
    its own network would buffer the request body."""
    host = host.lower()
    if host == "localhost" or host.endswith(".localhost") or ("." not in host and ":" not in host):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _bad_url() -> IQError:
    return IQError("iq-stream: set url (or IQ_STREAM_URL) to http(s)://host[:port].")


def _connector(url: str, trust_env: bool | None, verify: bool | ssl.SSLContext) -> Connector:
    parts = urlsplit(url)
    try:
        _ = parts.port  # raises for a port that is not a number
    except ValueError:
        raise _bad_url() from None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
        raise _bad_url()
    if trust_env is None:
        setting = (_env("IQ_STREAM_TRUST_ENV") or "").lower()
        if setting in _TRUE:
            trust_env = True
        elif setting in _FALSE:
            trust_env = False
        else:
            trust_env = not _private_host(parts.hostname)
    return Connector(url, trust_env, verify)


class IQClient:
    """The backend's client of an IQ-Stream server: it mints tokens, hands out tickets and publishes streams.

    It is given the app secret, so it belongs to the backend and never to a browser. It holds no connection and no
    thread until a publisher or a request needs one, so a worker pool may create it before it forks.

    Arguments default to the environment: ``IQ_STREAM_URL``, ``IQ_STREAM_APP_KEY``, ``IQ_STREAM_APP_SECRET``,
    ``IQ_STREAM_PUBLIC_URL``, ``IQ_STREAM_TRUST_ENV``. Durations are in seconds.
    """

    def __init__(
        self,
        url: str | None = None,
        app_key: str | None = None,
        app_secret: str | None = None,
        *,
        public_url: str | None = None,
        keepalive: float | None = 20.0,
        min_open: float = 3.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        on_error: Literal["report", "raise"] = "report",
        error_handler: Callable[[IQError], object] | None = None,
        retry_budget: float = 10.0,
        takeover_budget: float = 65.0,
        replay_frames: int = 512,
        replay_bytes: int = 4 << 20,
        queue_bytes: int = 8 << 20,
        trust_env: bool | None = None,
        verify: bool | ssl.SSLContext = True,
    ) -> None:
        resolved_url = (url or _env("IQ_STREAM_URL") or "").rstrip("/")
        connector = _connector(resolved_url, trust_env, verify)
        resolved_key = app_key if app_key is not None else _env("IQ_STREAM_APP_KEY") or ""
        if not isinstance(resolved_key, str) or not _APP_KEY.fullmatch(resolved_key):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise IQError(
                "iq-stream: set app_key (or IQ_STREAM_APP_KEY): 1 to 256 printable ASCII characters without spaces."
            )
        secret = app_secret if app_secret is not None else os.environ.get("IQ_STREAM_APP_SECRET", "")
        if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise IQError("iq-stream: set app_secret (or IQ_STREAM_APP_SECRET): at least 32 bytes.")
        if on_error not in ("report", "raise"):
            raise IQError("iq-stream: invalid option on_error: 'report' or 'raise'.")
        if error_handler is not None and not callable(error_handler):
            raise IQError("iq-stream: invalid option error_handler.")

        self._url = resolved_url
        self._public_url = (public_url or _env("IQ_STREAM_PUBLIC_URL") or resolved_url).rstrip("/")
        self._app_key = resolved_key
        self.__secret = secret.encode("utf-8")
        settings = Settings(
            keepalive=None if keepalive is None else _number("keepalive", keepalive, 0.001),
            min_open=_number("min_open", min_open, 0),
            max_frame_bytes=int(_number("max_frame_bytes", max_frame_bytes, 1, integer=True)),
            raise_errors=on_error == "raise",
            error_handler=error_handler,
            replay_frames=int(_number("replay_frames", replay_frames, 0, integer=True)),
            replay_bytes=int(_number("replay_bytes", replay_bytes, 0, integer=True)),
            queue_bytes=int(_number("queue_bytes", queue_bytes, 1, integer=True)),
            retry_budget=_number("retry_budget", retry_budget, 0),
            takeover_budget=_number("takeover_budget", takeover_budget, 0),
        )
        self._core = Core(
            settings=settings,
            connector=connector,
            mint=self._mint,
            sync_transport=connector.sync,
            async_transport=lambda: AsyncHTTP(connector),
        )

    @property
    def url(self) -> str:
        """How this backend reaches the server."""
        return self._url

    @property
    def public_url(self) -> str:
        """The address handed to browsers."""
        return self._public_url

    @property
    def app_key(self) -> str:
        return self._app_key

    @property
    def max_frame_bytes(self) -> int:
        """The limit one NDJSON line must fit, in bytes."""
        return self._core.settings.max_frame_bytes

    @property
    def trust_env(self) -> bool:
        """Whether httpx reads proxy settings from the environment for this client. The CA bundle that
        ``SSL_CERT_FILE`` or ``SSL_CERT_DIR`` names is used either way."""
        return self._core.connector.trust_env

    def __repr__(self) -> str:
        return f"IQClient(url={self._url!r}, app_key={self._app_key!r})"

    def new_stream_id(self) -> str:
        """128 random bits as base64url: 22 characters. Never reuse an id for a second answer."""
        return new_stream_id()

    def token(self, kind: Scope, stream_id: str, ttl: float = DEFAULT_TOKEN_TTL) -> str:
        """Mints a token. The ``sub`` token is what a browser subscribes with. The server refuses a ``ttl`` beyond
        its ``MAX_TOKEN_TTL_S`` (3600 by default)."""
        return self._mint(check_scope(kind), check_stream_id(stream_id), ttl)

    def verify(self, kind: Scope, token: str, stream_id: str) -> bool:
        """Checks a token as the server does. A backend that starts a run for a stream id sent by a page asks for
        the page's ``sub`` token too, and verifies it: the id then provably came from this backend."""
        scope = check_scope(kind)
        if not isinstance(token, str) or not isinstance(stream_id, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            return False
        return verify_token(self.__secret, scope, self._app_key, stream_id, token, time.time())

    def subscribe_url(self, stream_id: str, ttl: float = DEFAULT_TOKEN_TTL) -> str:
        """The subscribe URL on ``public_url``, with a ``sub`` token in its query."""
        return self.ticket(stream_id, ttl).url

    def ticket(self, stream_id: str | None = None, ttl: float = DEFAULT_TOKEN_TTL) -> Ticket:
        """What a browser needs to subscribe. Without a stream id, a new one is made."""
        if not self._public_url.startswith(("http://", "https://")):
            raise IQError("iq-stream: set public_url (or IQ_STREAM_PUBLIC_URL) to the address browsers use.")
        stream_id = check_stream_id(new_stream_id() if stream_id is None else stream_id)
        expires_at = self._expiry(ttl)
        token = sign(self.__secret, "sub", claims(self._app_key, stream_id, expires_at))
        return Ticket(stream_id, token, f"{self._public_url}/{stream_id}?token={token}", expires_at)

    def publisher(self, stream_id: str) -> Publisher:
        """The publisher of one stream, for synchronous code. Its request opens at once."""
        return Publisher(self._core, check_stream_id(stream_id))

    def apublisher(self, stream_id: str) -> AsyncPublisher:
        """The publisher of one stream, for asyncio. Called on a running loop; its request opens when the loop
        next runs."""
        return AsyncPublisher(self._core, check_stream_id(stream_id))

    def reserve(self, stream_id: str) -> PublishResult:
        """Takes a stream out of ``pending`` with one empty POST, for a run that waits in a queue: from then on the
        stream waits the idle timeout for its publisher, not the pending timeout."""
        stream_id = check_stream_id(stream_id)
        answer = self._core.sync_transport().post(
            stream_id, self._mint("pub", stream_id, PUB_TOKEN_TTL), b"", POST_TIMEOUT, Watch()
        )
        # publisher_active: the run has started already, which is as good as reserved.
        return self._accept(stream_id, answer, "publisher_active")

    async def areserve(self, stream_id: str) -> PublishResult:
        """``reserve()`` for asyncio."""
        stream_id = check_stream_id(stream_id)
        transport = self._core.async_transport()
        try:
            answer = await transport.post(
                stream_id, self._mint("pub", stream_id, PUB_TOKEN_TTL), b"", POST_TIMEOUT, Watch()
            )
        finally:
            await transport.aclose()
        return self._accept(stream_id, answer, "publisher_active")

    def close_stream(self, stream_id: str, outcome: Outcome = "fail") -> PublishResult:
        """Closes a stream from a process that never had its publisher, such as a job's failure handler, so that
        subscribers hear of it at once. A stream that is already closed is left as it is."""
        stream_id = check_stream_id(stream_id)
        program = close_stream_within(check_outcome(outcome), self._core.settings.retry_budget)
        transport = self._core.sync_transport()

        def carry_out(command: Command) -> Answer:
            if isinstance(command, Pause):
                time.sleep(command.seconds)
                return Answer(0)
            assert isinstance(command, Post)
            return transport.post(
                stream_id, self._mint("pub", stream_id, PUB_TOKEN_TTL), command.body, command.timeout, Watch()
            )

        return self._accept(stream_id, drive(program, carry_out), "stream_closed")

    async def aclose_stream(self, stream_id: str, outcome: Outcome = "fail") -> PublishResult:
        """``close_stream()`` for asyncio."""
        stream_id = check_stream_id(stream_id)
        program = close_stream_within(check_outcome(outcome), self._core.settings.retry_budget)
        transport: AsyncTransport = self._core.async_transport()

        async def carry_out(command: Command) -> Answer:
            if isinstance(command, Pause):
                await asyncio.sleep(command.seconds)
                return Answer(0)
            assert isinstance(command, Post)
            token = self._mint("pub", stream_id, PUB_TOKEN_TTL)
            return await transport.post(stream_id, token, command.body, command.timeout, Watch())

        try:
            answer = await adrive(program, carry_out)
        finally:
            await transport.aclose()
        return self._accept(stream_id, answer, "stream_closed")

    def _accept(self, stream_id: str, answer: Answer, harmless: str) -> PublishResult:
        if answer.status == 200 or (answer.status == 409 and answer.body.get("error") == harmless):
            return to_result(answer)
        raise refusal(stream_id, "the server refused the request", answer)

    def _expiry(self, ttl: float) -> int:
        return math.floor(time.time() + check_seconds("ttl", ttl, positive=True))

    def _mint(self, scope: Scope, stream_id: str, ttl: float) -> str:
        return sign(self.__secret, scope, claims(self._app_key, stream_id, self._expiry(ttl)))
