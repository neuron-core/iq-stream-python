"""The relay: iterate a graph's v2 stream, publish each part as one frame, yield it unchanged."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import langgraph.errors
from iq_stream import AsyncPublisherLike, FrameTooLarge, IQError, Outcome, PublisherLike

from ._encode import (
    CANCELLED_PART,
    DEFAULT_STREAM_MODE,
    Metadata,
    check_metadata,
    dumps,
    error_part,
    metadata_part,
    shape,
    truncated_messages_part,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel  # pyright: ignore[reportMissingTypeStubs]
    from langgraph.types import StreamPart

logger = logging.getLogger("iq_stream_langchain")

#: The close line of a cancelled run.
CancelledOutcome = Literal["fail", "end"]

# A drain request (langgraph 1.2) stops the run on purpose: a cancel, not a failure.
_DRAINED: tuple[type[BaseException], ...] = tuple(
    kind
    for kind in (getattr(langgraph.errors, "GraphDrained", None),)
    if isinstance(kind, type) and issubclass(kind, BaseException)
)


@dataclass(frozen=True)
class Oversize:
    """A part whose frame was over the publisher's ``max_frame_bytes``."""

    #: The part's ``type``: ``messages``, ``updates``, ...
    part_type: str
    #: Bytes of the frame that did not fit.
    size: int
    limit: int
    #: True when it was sent with its message content cut; False when it was not sent.
    truncated: bool


def stopped(pub: PublisherLike | AsyncPublisherLike) -> bool:
    """Whether the publisher takes no more frames. One that gave up with ``on_error="raise"`` raises its error here:
    that option fails the run, and the relay must not keep it quiet by sending nothing."""
    if pub.alive:
        return False
    if pub.error is not None:
        pub.send("")  # raises with on_error="raise"; a no-op on a publisher that gave up otherwise
    return True


class _Run:
    """One run's publishing: the publisher, the options, and what was reported once already."""

    def __init__(
        self,
        pub: PublisherLike | AsyncPublisherLike,
        stream_mode: Sequence[str],
        also_yield: Sequence[str],
        metadata: Metadata,
        expose_errors: bool,
        cancelled: CancelledOutcome,
        on_oversize: Callable[[Oversize], None] | None,
    ) -> None:
        check_metadata(metadata)
        if cancelled not in ("fail", "end"):
            raise ValueError('cancelled is "fail" or "end".')
        if isinstance(stream_mode, str) or isinstance(also_yield, str):
            raise ValueError("stream_mode and also_yield are sequences of mode names.")
        self.pub = pub
        self.published = frozenset(stream_mode)
        self.modes = list(dict.fromkeys([*stream_mode, *also_yield]))
        self.metadata: Metadata = metadata
        self.expose_errors = expose_errors
        self.cancelled: CancelledOutcome = cancelled
        self.on_oversize = on_oversize
        self.unencodable_reported = False
        self.oversize_reported = False

    def send(self, data: str) -> None:
        """Sends a part the adapter made. They are small: one that is too large is dropped."""
        if self.pub.alive:
            with contextlib.suppress(FrameTooLarge):
                self.pub.send(data)

    def publish(self, part: Mapping[str, Any]) -> None:
        # A publisher that gave up is not fed: nothing is encoded for the rest of the run, which goes on.
        if part["type"] not in self.published or stopped(self.pub):
            return
        try:
            shaped = shape(part, metadata=self.metadata, expose_errors=self.expose_errors)
            data = dumps(shaped)
        except Exception:
            # One frame is lost, never the run.
            if not self.unencodable_reported:
                self.unencodable_reported = True
                logger.warning(
                    "IQ stream %s: a %s part could not be encoded and was not sent; later ones are not reported.",
                    self.pub.stream_id,
                    part["type"],
                )
            return
        try:
            self.pub.send(data)
        except FrameTooLarge as error:
            self._oversize(shaped, error)

    def _oversize(self, shaped: Mapping[str, Any], error: FrameTooLarge) -> None:
        limit = self.pub.max_frame_bytes
        cut = truncated_messages_part(shaped, lambda data: self.pub.frame_size(data) <= limit)
        if cut is not None:
            self.pub.send(cut)
        report = Oversize(str(shaped["type"]), error.size, error.limit, truncated=cut is not None)
        if self.on_oversize is not None:
            self.on_oversize(report)
        elif not self.oversize_reported:
            self.oversize_reported = True
            logger.warning(
                "IQ stream %s: a %s part of %d bytes is over the frame limit of %d bytes and was %s; "
                "later ones are not reported.",
                self.pub.stream_id,
                report.part_type,
                report.size,
                report.limit,
                "truncated" if report.truncated else "not sent",
            )

    def fail(self, error: BaseException) -> None:
        """The run raised: its ``error`` part, before the exception closes the stream with ``fail``."""
        if isinstance(error, IQError):
            return  # the publisher's own error, with on_error="raise": nothing more can be sent
        if not self.pub.alive:
            return
        with contextlib.suppress(IQError):
            try:
                self.pub.send(error_part(error, expose=self.expose_errors))
            except FrameTooLarge:
                self.pub.send(error_part(error))

    def cancel(self) -> None:
        with contextlib.suppress(IQError):
            self.send(CANCELLED_PART)
            cast("PublisherLike", self.pub).close(self.cancelled)

    async def acancel(self) -> None:
        with contextlib.suppress(IQError):
            self.send(CANCELLED_PART)
            await aclose_publisher(self.pub, self.cancelled)


async def aclose_publisher(pub: AsyncPublisherLike | PublisherLike, outcome: Outcome) -> None:
    """Closes either kind of publisher from asyncio code."""
    if inspect.iscoroutinefunction(pub.close):
        await cast("AsyncPublisherLike", pub).close(outcome)
    else:
        # A synchronous publisher's close waits for the server: not on the loop's thread.
        await asyncio.to_thread(cast("PublisherLike", pub).close, outcome)


def relay(
    pub: PublisherLike,
    graph: Pregel[Any, Any, Any, Any],
    input: Any,
    config: RunnableConfig | None = None,
    *,
    stream_mode: Sequence[str] = DEFAULT_STREAM_MODE,
    also_yield: Sequence[str] = (),
    metadata: Metadata = "slim",
    expose_errors: bool = False,
    cancelled: CancelledOutcome = "fail",
    on_oversize: Callable[[Oversize], None] | None = None,
    **stream_kwargs: Any,
) -> Generator[StreamPart[Any, Any], None, None]:
    """Runs ``graph.stream(input, config, subgraphs=True, version="v2")``, publishes each part whose mode is in
    ``stream_mode`` as one frame, and yields every part unchanged.

    The first frame is a ``metadata`` part with the config's ``run_id`` and ``thread_id``. ``also_yield`` modes are
    requested from the graph and yielded, not published. A run that raises publishes an ``error`` part and the
    exception goes on, so that the publisher's ``with`` block closes the stream with ``fail``. A run that is
    cancelled (the loop is left early, a ``KeyboardInterrupt``, a drain request) publishes a ``Cancelled`` error
    part and closes the publisher with ``cancelled`` itself. A run that finishes, or pauses on an interrupt, is left
    to the ``with`` block, which closes with ``end`` after the application has saved the answer.

    ``stream_kwargs`` go to ``graph.stream()``: ``context``, ``durability``, ``interrupt_before``, ``control``, ...
    """
    if inspect.iscoroutinefunction(pub.close):
        raise TypeError("relay() takes a synchronous publisher; give an AsyncPublisher to arelay().")
    run = _Run(pub, stream_mode, also_yield, metadata, expose_errors, cancelled, on_oversize)
    return _relay(run, graph, input, config, stream_kwargs)


def _relay(
    run: _Run, graph: Pregel[Any, Any, Any, Any], input: Any, config: Any, stream_kwargs: dict[str, Any]
) -> Generator[StreamPart[Any, Any], None, None]:
    run.send(metadata_part(config))
    try:
        # The overloads of stream() do not take a list of plain strings, nor keyword arguments passed along.
        parts: Iterator[StreamPart[Any, Any]] = cast("Any", graph).stream(
            input, config, stream_mode=run.modes, subgraphs=True, version="v2", **stream_kwargs
        )
        try:
            for part in parts:
                run.publish(part)
                yield part
        finally:
            # The run stops now when the loop is left, not when the generator is collected.
            close = getattr(parts, "close", None)
            if callable(close):
                close()
    except _DRAINED:
        run.cancel()
        raise
    except Exception as error:
        run.fail(error)
        raise
    except BaseException:  # GeneratorExit when the loop is left early, KeyboardInterrupt
        run.cancel()
        raise


def arelay(
    pub: AsyncPublisherLike | PublisherLike,
    graph: Pregel[Any, Any, Any, Any],
    input: Any,
    config: RunnableConfig | None = None,
    *,
    stream_mode: Sequence[str] = DEFAULT_STREAM_MODE,
    also_yield: Sequence[str] = (),
    metadata: Metadata = "slim",
    expose_errors: bool = False,
    cancelled: CancelledOutcome = "fail",
    on_oversize: Callable[[Oversize], None] | None = None,
    **stream_kwargs: Any,
) -> AsyncGenerator[StreamPart[Any, Any], None]:
    """``relay()`` over ``graph.astream()``. A cancelled task publishes the ``Cancelled`` part and closes the
    stream as a cancelled run, like a loop that is left early under ``contextlib.aclosing()``.

    ``pub`` is usually an ``AsyncPublisher``. A synchronous ``Publisher`` fits too, for graphs whose nodes block the
    loop and would starve the asynchronous publisher's keepalive: ``send()`` is the same call.
    """
    run = _Run(pub, stream_mode, also_yield, metadata, expose_errors, cancelled, on_oversize)
    return _arelay(run, graph, input, config, stream_kwargs)


async def _arelay(
    run: _Run, graph: Pregel[Any, Any, Any, Any], input: Any, config: Any, stream_kwargs: dict[str, Any]
) -> AsyncGenerator[StreamPart[Any, Any], None]:
    run.send(metadata_part(config))
    try:
        parts: AsyncIterator[StreamPart[Any, Any]] = cast("Any", graph).astream(
            input, config, stream_mode=run.modes, subgraphs=True, version="v2", **stream_kwargs
        )
        try:
            async for part in parts:
                run.publish(part)
                yield part
        finally:
            # An async generator is not closed when a loop over it is left: the run would go on until it is collected.
            aclose = getattr(parts, "aclose", None)
            if callable(aclose):
                await cast("Callable[[], Awaitable[None]]", aclose)()
    except _DRAINED:
        await run.acancel()
        raise
    except Exception as error:
        run.fail(error)
        raise
    except BaseException:  # CancelledError, GeneratorExit from aclose(), KeyboardInterrupt
        await run.acancel()
        raise


def _final_state(part: Mapping[str, Any], final: Any) -> Any:
    return part["data"] if part["type"] == "values" and not part["ns"] else final


def publish_run(
    pub: PublisherLike,
    graph: Pregel[Any, Any, Any, Any],
    input: Any,
    config: RunnableConfig | None = None,
    *,
    stream_mode: Sequence[str] = DEFAULT_STREAM_MODE,
    also_yield: Sequence[str] = (),
    metadata: Metadata = "slim",
    expose_errors: bool = False,
    cancelled: CancelledOutcome = "fail",
    on_oversize: Callable[[Oversize], None] | None = None,
    **stream_kwargs: Any,
) -> Any:
    """Relays the run to its end and returns the final state of the root graph (its last ``values``, which are
    requested from the graph and not published). Save the answer after it returns, inside the publisher's block."""
    final: Any = None
    parts = relay(
        pub,
        graph,
        input,
        config,
        stream_mode=stream_mode,
        also_yield=[*also_yield, "values"],
        metadata=metadata,
        expose_errors=expose_errors,
        cancelled=cancelled,
        on_oversize=on_oversize,
        **stream_kwargs,
    )
    for part in parts:
        final = _final_state(part, final)
    return final


async def apublish_run(
    pub: AsyncPublisherLike | PublisherLike,
    graph: Pregel[Any, Any, Any, Any],
    input: Any,
    config: RunnableConfig | None = None,
    *,
    stream_mode: Sequence[str] = DEFAULT_STREAM_MODE,
    also_yield: Sequence[str] = (),
    metadata: Metadata = "slim",
    expose_errors: bool = False,
    cancelled: CancelledOutcome = "fail",
    on_oversize: Callable[[Oversize], None] | None = None,
    **stream_kwargs: Any,
) -> Any:
    """``publish_run()`` over ``arelay()``. A cancelled task closes the stream as a cancelled run."""
    final: Any = None
    parts = arelay(
        pub,
        graph,
        input,
        config,
        stream_mode=stream_mode,
        also_yield=[*also_yield, "values"],
        metadata=metadata,
        expose_errors=expose_errors,
        cancelled=cancelled,
        on_oversize=on_oversize,
        **stream_kwargs,
    )
    async with contextlib.aclosing(parts):
        async for part in parts:
            final = _final_state(part, final)
    return final
