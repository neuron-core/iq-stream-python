"""``IQStream``: a context manager around one Haystack run that publishes its ``StreamingChunk`` objects."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping
from types import TracebackType
from typing import Literal, Self, overload

from ag_ui.core import BaseEvent
from haystack.dataclasses import StreamingChunk
from iq_stream import FrameTooLarge, IQClient, Outcome, PublisherLike, PublishResult, Ticket

from .converters import AGUIConverter, AISDKConverter, ChunkConverter, ConverterOptions, Frame, NativeConverter
from .converters.agui import encode_event
from .converters.base import dumps, source_key

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
    ``cancelled_close``. The run's exception goes on unchanged. Publishing never breaks the run: an error is reported
    once to ``on_error`` (else logged as a warning), publishing stops, the stream is closed with ``fail``, and the
    run is read to its end.

    ``client`` is an ``IQClient``, whose publisher the block opens, or a publisher that is already open
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
        cancelled_close: Outcome = "fail",
        close_timeout: float = 10.0,
        on_error: Callable[[Exception], object] | None = None,
    ) -> None:
        if isinstance(client, IQClient):
            self._client: IQClient | None = client
            self._given: PublisherLike | None = None
            stream_id = client.new_stream_id() if stream_id is None else stream_id
        else:
            self._client = None
            self._given = client
            if stream_id is not None and stream_id != client.stream_id:
                raise ValueError("stream_id differs from the publisher's.")
            stream_id = client.stream_id
        if not isinstance(stream_id, str) or not _STREAM_ID.fullmatch(stream_id):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("stream_id must be 16 to 128 characters of A-Z a-z 0-9 _ -")
        if isinstance(protocol, str) and protocol not in ("ag-ui", "ai-sdk", "haystack"):
            raise ValueError("protocol must be 'ag-ui', 'ai-sdk', 'haystack' or a ChunkConverter.")
        if cancelled_close not in ("end", "fail"):
            raise ValueError("cancelled_close must be 'end' or 'fail'.")
        if max_tool_result_bytes < 1:
            raise ValueError("max_tool_result_bytes must be at least 1.")
        self._stream_id = stream_id
        self._protocol = protocol
        self._thread_id = thread_id or stream_id
        self._run_id = run_id or stream_id
        self._include_tool_results = include_tool_results
        self._max_tool_result_bytes = max_tool_result_bytes
        self._expose_errors = expose_errors
        self._cancelled_close: Outcome = cancelled_close
        self._close_timeout = close_timeout
        self._on_error = on_error

        # Haystack calls back from tool threads and from asyncio.to_thread workers: one lock orders every frame.
        self._lock = threading.Lock()
        self._state: Literal["new", "open", "closing", "closed"] = "new"
        self._publisher: PublisherLike | None = None
        self._converter: ChunkConverter | None = None
        self._error: Exception | None = None
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
    def error(self) -> Exception | None:
        """Why publishing stopped: this adapter's error, or the publisher's. The run itself goes on."""
        if self._error is not None:
            return self._error
        return None if self._publisher is None else self._publisher.error

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
        """The streaming callback for ``run()``, and for ``run_async()`` with a chat generator that has no
        ``run_async``. Thread-safe; never raises, never blocks on the network."""
        if self._state != "open":
            return
        try:
            with self._lock:
                if self._publishing():
                    assert self._converter is not None
                    self._send(self._converter.convert(chunk, source_key()))
        except Exception as error:
            self._fail(error)

    async def acallback(self, chunk: StreamingChunk) -> None:
        """The streaming callback for ``run_async()``. It awaits nothing: frames are queued."""
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
        Returns whether it was queued. ``FrameTooLarge`` is raised for a frame over the limit."""
        if isinstance(payload, BaseEvent):
            data = encode_event(payload)
        elif isinstance(payload, str):
            data = payload
        else:
            data = dumps(dict(payload))
        return self._emit(lambda: [Frame(data, event)])

    def custom(self, name: str, value: object) -> bool:
        """Publishes a named event of the application's own in the stream's protocol: an AG-UI ``CUSTOM`` event, an
        AI SDK ``data-<name>`` chunk, or ``{"type": "custom"}``. A human-in-the-loop strategy uses it to publish its
        request before it blocks. Returns whether it was queued."""
        return self._emit(lambda: self._require_converter().custom(name, value))

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
            with self._lock:
                self._send(self._converter.start())
        except Exception as error:
            self._fail(error)
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

    def _require_converter(self) -> ChunkConverter:
        if self._converter is None:
            raise RuntimeError("The stream is not open.")
        return self._converter

    def _publishing(self) -> bool:
        return (
            self._state == "open"
            and self._error is None
            and self._publisher is not None
            and self._publisher.alive
            and self._converter is not None
        )

    def _send(self, frames: Iterable[Frame]) -> None:
        """Queues frames; the caller holds the lock. Stops at the first refusal: the publisher has given up."""
        assert self._publisher is not None
        for frame in frames:
            if not self._publisher.send(frame.data, frame.event):
                return

    def _emit(self, frames: Callable[[], Iterable[Frame]]) -> bool:
        with self._lock:
            if not self._publishing():
                return False
            assert self._publisher is not None
            built = list(frames())
            for frame in built:  # nothing is sent unless every frame fits
                size = self._publisher.frame_size(frame.data, frame.event)
                if size > self._publisher.max_frame_bytes:
                    raise FrameTooLarge(size, self._publisher.max_frame_bytes)
            self._send(built)
            return self._publisher.alive

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._error is not None:
                return
            self._error = error
        if self._on_error is None:
            logger.warning(
                "IQ stream %s: publishing stopped after %s; the run goes on.", self._stream_id, type(error).__name__
            )
            return
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
            publishing = self._publishing()
            self._state = "closing"
        publisher = self._publisher
        if publisher is None:
            self._state = "closed"
            return
        if publishing:
            assert self._converter is not None
            try:
                with self._lock:
                    if exc is None:
                        self._send(self._converter.finish())
                    elif cancelled:
                        self._send(self._converter.cancelled())
                    else:
                        self._send(self._converter.error(exc))
            except Exception as error:
                self._fail(error)
        if self._error is not None or (exc is not None and not cancelled):
            outcome: Outcome = "fail"
        else:
            outcome = self._cancelled_close if cancelled else "end"
        try:
            self._result = publisher.close(outcome, self._close_timeout)
        except Exception as error:
            self._fail(error)
        finally:
            self._state = "closed"
