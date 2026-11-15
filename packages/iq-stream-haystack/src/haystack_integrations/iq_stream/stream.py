"""``IQStream``: a context manager around one Haystack run that publishes its ``StreamingChunk`` objects."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import math
import re
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Literal, overload

from ag_ui.core import BaseEvent
from haystack.dataclasses import StreamingChunk
from iq_stream import FrameTooLarge, IQClient, IQError, Outcome, PublisherLike, PublishResult, Ticket

from .converters import AGUIConverter, AISDKConverter, ChunkConverter, ConverterOptions, Frame, NativeConverter
from .converters.agui import encode_event
from .converters.base import dumps, source_key

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger("haystack_integrations.iq_stream")

ProtocolName = Literal["ag-ui", "ai-sdk", "haystack"]

_STREAM_ID = re.compile(r"[A-Za-z0-9_-]{16,128}")
_ACTIVE: contextvars.ContextVar[IQStream | None] = contextvars.ContextVar("iq_stream_haystack", default=None)


def current_stream() -> IQStream | None:
    """The ``IQStream`` whose block is active in this context (thread, task, or a tool thread Haystack started from
    it), or ``None``."""
    return _ACTIVE.get()


class IQStream:
    """Publishes one Haystack run to an IQ-Stream server.

    Create it where the run happens and enter it around the run; hand Haystack ``callback`` (for ``run()``) or
    ``acallback`` (for ``run_async()``) as the ``streaming_callback``, and save the run's result inside the block::

        with IQStream(iq, stream_id) as stream:
            result = agent.run(messages, streaming_callback=stream.callback)
            save(result["messages"])  # before the close line

    Entering opens the publisher and sends the protocol's start event. Leaving sends the protocol's final event and
    closes the stream with ``end``, or with ``fail`` when the block raised; a cancelled run closes with
    ``close_on_cancel``. The run's exception goes on unchanged.

    Publishing never breaks the run. A chunk that cannot be converted, or a frame of the application's own that
    cannot be encoded or is too large, is skipped and reported to ``on_error`` (else logged once per kind), and the
    stream goes on. When the publisher gives up (the core's ``IQError``) or fails, publishing stops, ``on_error``
    hears of it once, the stream closes with ``fail``, and the run is read to its end. With a client opened with
    ``on_error="raise"``, leaving the block then raises the publisher's ``IQError``, unless the block raised already.

    ``client`` is an ``IQClient``, whose publisher the block opens, or a synchronous publisher that is already open
    (``iq_stream.PublisherLike``, such as the testing kit's ``RecordingPublisher``). Creating the object opens
    nothing, so the process that hands out the ticket may create one to learn the stream id, and the worker that
    runs the agent another with the same id.
    """

    def __init__(
        self,
        client: IQClient | PublisherLike,
        stream_id: str | None = None,
        *,
        protocol: ProtocolName | ChunkConverter = "ag-ui",
        thread_id: str | None = None,
        run_id: str | None = None,
        include_tool_results: bool = True,
        max_tool_result_bytes: int = 32_768,
        expose_errors: bool = False,
        close_on_cancel: Outcome = "fail",
        close_timeout: float = 10.0,
        on_error: Callable[[IQError], object] | None = None,
    ) -> None:
        if isinstance(client, IQClient):
            self._client: IQClient | None = client
            self._given: PublisherLike | None = None
            stream_id = client.new_stream_id() if stream_id is None else stream_id
        else:
            if inspect.iscoroutinefunction(getattr(client, "close", None)):
                # Haystack calls back from worker threads, where an asyncio publisher cannot be used.
                raise TypeError(
                    "IQStream takes a synchronous publisher: client.publisher(), or RecordingPublisher in tests."
                )
            self._client = None
            self._given = client
            if stream_id is not None and stream_id != client.stream_id:
                raise ValueError("stream_id differs from the publisher's.")
            stream_id = client.stream_id
        if not isinstance(stream_id, str) or not _STREAM_ID.fullmatch(stream_id):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("stream_id must be 16 to 128 characters of A-Z a-z 0-9 _ -")
        if isinstance(protocol, str) and protocol not in ("ag-ui", "ai-sdk", "haystack"):
            raise ValueError("protocol must be 'ag-ui', 'ai-sdk', 'haystack' or a ChunkConverter.")
        if close_on_cancel not in ("end", "fail"):
            raise ValueError("close_on_cancel must be 'end' or 'fail'.")
        if max_tool_result_bytes < 1:
            raise ValueError("max_tool_result_bytes must be at least 1.")
        if (
            isinstance(close_timeout, bool)
            or not isinstance(close_timeout, int | float)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not math.isfinite(close_timeout)
            or close_timeout <= 0
        ):
            # 0 would not wait for the server's answer at all: every stream would end failed with close_timeout.
            raise ValueError("close_timeout must be a finite number of seconds above 0.")
        self._stream_id = stream_id
        self._protocol = protocol
        self._thread_id = thread_id or stream_id
        self._run_id = run_id or stream_id
        self._include_tool_results = include_tool_results
        self._max_tool_result_bytes = max_tool_result_bytes
        self._expose_errors = expose_errors
        self._close_on_cancel: Outcome = close_on_cancel
        self._close_timeout = close_timeout
        self._on_error = on_error

        # Haystack calls back from tool threads and from asyncio.to_thread workers: one lock orders every frame.
        self._lock = threading.Lock()
        self._state: Literal["new", "open", "closing", "closed"] = "new"
        self._publisher: PublisherLike | None = None
        self._converter: ChunkConverter | None = None
        self._stopped: IQError | None = None  # why publishing stopped on this side
        self._heard = False  # on_error was told that the publisher gave up
        self._logged: set[str] = set()
        self._token: contextvars.Token[IQStream | None] | None = None
        self._result: PublishResult | None = None

    # -- what the application reads --------------------------------------------------------------------------------

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def publisher(self) -> PublisherLike | None:
        """The publisher, once the block was entered."""
        return self._publisher

    @property
    def error(self) -> IQError | None:
        """Why publishing stopped: the publisher's ``IQError`` when it gave up, else this adapter's when the publisher
        failed (code ``publish_failed``). The run itself goes on."""
        if self._publisher is not None and self._publisher.error is not None:
            return self._publisher.error
        return self._stopped

    @property
    def failed(self) -> bool:
        """Whether publishing stopped before the run ended."""
        return self.error is not None

    @property
    def result(self) -> PublishResult | None:
        """The server's answer to the close line, once the block was left."""
        return self._result

    def ticket(self, ttl: float = 3600) -> Ticket:
        """What the browser needs to subscribe: ``stream_id``, the ``sub`` token, the URL and its expiry."""
        if self._client is None:
            raise RuntimeError("ticket() needs the IQClient: this stream was given a publisher.")
        return self._client.ticket(self._stream_id, ttl)

    # -- the callbacks handed to Haystack ----------------------------------------------------------------------------

    def callback(self, chunk: StreamingChunk) -> None:
        """The streaming callback for ``run()``, and for ``run_async()`` whenever a generator that gets it has no
        ``run_async``. Thread-safe; never raises, never blocks on the network."""
        if self._state == "open":
            self._publish(lambda converter: converter.convert(chunk, source_key()), "a chunk")

    async def acallback(self, chunk: StreamingChunk) -> None:
        """The streaming callback for ``run_async()`` when every generator that gets it has ``run_async``: Haystack
        refuses a coroutine callback where it runs a generator in a thread. It awaits nothing: frames are queued."""
        self.callback(chunk)

    @overload
    def tee(
        self, other: Callable[[StreamingChunk], Awaitable[None]]
    ) -> Callable[[StreamingChunk], Awaitable[None]]: ...

    @overload
    def tee(self, other: Callable[[StreamingChunk], None]) -> Callable[[StreamingChunk], None]: ...

    def tee(
        self, other: Callable[[StreamingChunk], Awaitable[None]] | Callable[[StreamingChunk], None]
    ) -> Callable[[StreamingChunk], Awaitable[None]] | Callable[[StreamingChunk], None]:
        """A callback that publishes each chunk and then calls ``other``, the application's own callback: a callback
        given at run time replaces the one a component was built with. Async when ``other`` is."""
        if inspect.iscoroutinefunction(other):
            second = other

            async def both_async(chunk: StreamingChunk) -> None:
                self.callback(chunk)
                await second(chunk)

            return both_async

        def both(chunk: StreamingChunk) -> None:
            self.callback(chunk)
            other(chunk)

        return both

    # -- frames of the application's own -----------------------------------------------------------------------------

    def emit(self, payload: BaseEvent | Mapping[str, object] | str, event: str | None = None) -> bool:
        """Publishes one frame as given, in order with the run's: an AG-UI event, a mapping (as JSON) or a string.
        Returns whether it was queued. It never raises: a frame that cannot be encoded or is over the limit is dropped
        and reported to ``on_error`` (else logged), and the stream goes on."""

        def frames(converter: ChunkConverter) -> list[Frame]:
            if isinstance(payload, BaseEvent):
                data = encode_event(payload)
            elif isinstance(payload, str):
                data = payload
            else:
                data = dumps(dict(payload))
            return [Frame(data, event)]

        return self._publish(frames, "a frame of the application's", whole=True)

    def custom(self, name: str, value: object) -> bool:
        """Publishes a named event of the application's own in the stream's protocol: an AG-UI ``CUSTOM`` event, an
        AI SDK ``data-<name>`` chunk, or ``{"type": "custom"}``; what JSON cannot hold is written as its ``str()``, a
        NaN or an infinity as ``null``. A human-in-the-loop strategy uses it to publish its request before it blocks.
        Returns whether it was queued, and never raises, as ``emit``."""
        return self._publish(lambda converter: converter.custom(name, value), "a custom event", whole=True)

    # -- lifecycle --------------------------------------------------------------------------------------------------

    def __enter__(self) -> Self:
        with self._lock:
            if self._state != "new":
                raise RuntimeError("An IQStream serves one run: create a new one for the next run.")
            self._state = "open"
        self._token = _ACTIVE.set(self)
        try:
            publisher = self._given if self._given is not None else self._open()
            self._publisher = publisher
            self._converter = self._make_converter(publisher)
        except Exception as error:
            self._stop(error)
        else:
            self._publish(lambda converter: converter.start(), "the start of the run")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        self._leave()
        self._finish(exc)
        return False

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        self._leave()
        # close() waits for the server: off the event loop, and carried through even when this task is cancelled.
        await asyncio.shield(asyncio.to_thread(self._finish, exc))
        return False

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        # Pipeline.run deep-copies its inputs, and with them the object a bound-method callback belongs to.
        return self

    def __copy__(self) -> Self:
        return self

    def __repr__(self) -> str:
        return f"IQStream(stream_id={self._stream_id!r}, state={self._state!r})"

    # -- internals ---------------------------------------------------------------------------------------------------

    def _open(self) -> PublisherLike:
        assert self._client is not None
        return self._client.publisher(self._stream_id)

    def _make_converter(self, publisher: PublisherLike) -> ChunkConverter:
        if not isinstance(self._protocol, str):
            return self._protocol
        options = ConverterOptions(
            include_tool_results=self._include_tool_results,
            max_tool_result_bytes=self._max_tool_result_bytes,
            expose_errors=self._expose_errors,
            max_frame_bytes=publisher.max_frame_bytes,
            measure=publisher.frame_size,
        )
        if self._protocol == "ag-ui":
            return AGUIConverter(thread_id=self._thread_id, run_id=self._run_id, **options)
        if self._protocol == "ai-sdk":
            return AISDKConverter(message_id=self._run_id, **options)
        return NativeConverter(**options)

    def _publish(
        self,
        build: Callable[[ChunkConverter], Iterable[Frame]],
        what: str,
        *,
        whole: bool = False,
        closing: bool = False,
    ) -> bool:
        """Builds frames with the converter and queues them, under the lock, and returns whether they were queued.
        Never raises. What cannot be built, or a frame that does not fit, is skipped and reported, and publishing
        goes on; ``whole`` frames are measured first, so that none is sent unless all fit. A publisher that fails
        stops publishing."""
        skipped: IQError | None = None
        failure: Exception | None = None
        sent = False
        with self._lock:
            if self._publishing(closing):
                assert self._converter is not None
                try:
                    frames = list(build(self._converter))
                    if whole:
                        self._check_fit(frames)
                    sent = self._send(frames)
                except FrameTooLarge as error:
                    skipped = error
                except _PublisherFailed as error:
                    failure = error.cause
                except Exception as cause:
                    skipped = IQError(
                        f"IQ stream {self._stream_id}: {what} could not be encoded ({type(cause).__name__}); "
                        "it was skipped.",
                        stream_id=self._stream_id,
                        code="encode_failed",
                    )
                    skipped.__cause__ = cause
        if skipped is not None:
            self._report(skipped)
        elif failure is not None:
            self._stop(failure)
        elif not sent:
            self._note_gave_up()
        return sent

    def _publishing(self, closing: bool = False) -> bool:
        return (
            self._state == ("closing" if closing else "open")
            and self._stopped is None
            and self._publisher is not None
            and self._publisher.alive
            and self._converter is not None
        )

    def _check_fit(self, frames: list[Frame]) -> None:
        assert self._publisher is not None
        for frame in frames:
            size = self._publisher.frame_size(frame.data, frame.event)
            if size > self._publisher.max_frame_bytes:
                raise FrameTooLarge(size, self._publisher.max_frame_bytes)

    def _send(self, frames: Iterable[Frame]) -> bool:
        """Queues frames; the caller holds the lock. Returns False once the publisher has given up. A frame the
        publisher refuses (too large, invalid) raises as it is; a publisher that fails raises ``_PublisherFailed``."""
        publisher = self._publisher
        assert publisher is not None
        for frame in frames:
            try:
                alive = publisher.send(frame.data, frame.event)
            except (FrameTooLarge, ValueError):
                raise
            except Exception as error:
                if error is publisher.error:  # with on_error="raise": the error the publisher gave up on
                    return False
                raise _PublisherFailed(error) from error
            if not alive:
                return False
        return True

    def _stop(self, cause: Exception) -> None:
        """Publishing stops: the publisher failed, or could not be opened. Reported once."""
        error = IQError(
            f"IQ stream {self._stream_id}: publishing stopped after {type(cause).__name__}; the run goes on.",
            stream_id=self._stream_id,
            code="publish_failed",
        )
        error.__cause__ = cause
        with self._lock:
            if self._stopped is not None:
                return
            self._stopped = error
        self._report(error)

    def _note_gave_up(self) -> None:
        """Tells ``on_error``, once, that the core's publisher gave up (unreachable, stream too large, ...). The core
        logged it already, unless its client raises it: then leaving the block raises it, or logs it (``_finish``)."""
        publisher = self._publisher
        if publisher is None or publisher.alive or publisher.error is None:
            return
        with self._lock:
            if self._heard:
                return
            self._heard = True
        if self._on_error is not None:
            self._call_on_error(publisher.error)

    def _report(self, error: IQError) -> None:
        """Hands ``error`` to ``on_error``, else logs it once per kind, without frame data."""
        if self._on_error is not None:
            self._call_on_error(error)
            return
        kind = error.code or type(error).__name__
        with self._lock:
            if kind in self._logged:
                return
            self._logged.add(kind)
        if isinstance(error, FrameTooLarge):
            logger.warning(
                "IQ stream %s: a frame of %d bytes was not published; the limit is %d.",
                self._stream_id,
                error.size,
                error.limit,
            )
        else:
            logger.warning("%s", error)

    def _call_on_error(self, error: IQError) -> None:
        assert self._on_error is not None
        try:
            self._on_error(error)
        except Exception:
            logger.exception("IQ stream %s: on_error raised.", self._stream_id)

    def _leave(self) -> None:
        if self._token is None:
            return
        token, self._token = self._token, None
        try:
            _ACTIVE.reset(token)
        except ValueError:  # left in another context than the one it was entered in
            _ACTIVE.set(None)

    def _finish(self, exc: BaseException | None) -> None:
        cancelled = isinstance(exc, asyncio.CancelledError)
        with self._lock:
            if self._state != "open":
                return
            self._state = "closing"
        publisher = self._publisher
        if publisher is None:
            self._state = "closed"
            return
        if exc is None:
            self._publish(lambda converter: converter.finish(), "the end of the run", closing=True)
        elif cancelled:
            self._publish(lambda converter: converter.cancelled(), "the end of the run", closing=True)
        else:
            self._publish(lambda converter: converter.error(exc), "the run's error", closing=True)
        if self._stopped is not None or publisher.error is not None or (exc is not None and not cancelled):
            outcome: Outcome = "fail"
        else:
            outcome = self._close_on_cancel if cancelled else "end"
        try:
            self._result = publisher.close(outcome, self._close_timeout)
        except Exception as error:
            if error is not publisher.error:  # with on_error="raise", close() raises the error it gave up on: below
                self._stop(error)
                if publisher.alive:  # a failed stream is closed, never left open
                    with contextlib.suppress(Exception):
                        self._result = publisher.close("fail")
        finally:
            self._state = "closed"
        self._note_gave_up()
        gave_up = publisher.error
        if gave_up is not None and publisher.on_error == "raise":
            if exc is None:
                raise gave_up
            if self._on_error is None:  # the core logs only in report mode, and the block's exception goes on
                logger.warning("%s; the block's own exception goes on.", gave_up)


class _PublisherFailed(Exception):
    def __init__(self, cause: Exception) -> None:
        super().__init__(type(cause).__name__)
        self.cause = cause
