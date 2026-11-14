"""How events reach a stream: a scoped stream sink, or handlers on CrewAI's global event bus as a fallback."""

from __future__ import annotations

import contextlib
import contextvars
import heapq
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal, Protocol

from crewai.events.base_events import BaseEvent

from ._snapshot import EVENT_FIELDS

Deliver = Callable[[object], None]
Snapshot = Callable[[object, BaseEvent], "object | None"]

try:
    from crewai.events import stream_context as _sinks
except ImportError:  # pragma: no cover - CrewAI before 1.15.2, or a future one without it
    _sinks = None


class Transport(Protocol):
    def attach(self) -> None:
        """Starts delivering the events emitted in the current context and its copies."""
        ...

    def detach(self) -> None:
        """Stops taking new events; called in the context ``attach`` was called in."""
        ...

    def settle(self) -> None:
        """Waits until every event taken has been delivered. May block."""
        ...

    @property
    def blocking(self) -> bool:
        """Whether ``settle`` can block."""
        ...


def choose(transport: Literal["auto", "sink", "bus"]) -> Literal["sink", "bus"]:
    if transport == "bus" or (transport == "auto" and _sinks is None):
        return "bus"
    if _sinks is None:
        raise ImportError("transport='sink' needs crewai.events.stream_context (CrewAI 1.15.2 or later).")
    return "sink"


class SinkTransport:
    """A sink registered with ``add_stream_sink``. CrewAI calls it synchronously, in the emitting thread, for every
    event emitted in the context it was registered in or in any copy of it, before any handler runs: nothing can
    overtake an event on its way here, and nothing is left to flush when the block exits."""

    blocking = False

    def __init__(self, snapshot: Snapshot, deliver: Deliver) -> None:
        self._snapshot = snapshot
        self._deliver = deliver
        self._token: contextvars.Token[Any] | None = None

    def _sink(self, source: object, event: BaseEvent) -> None:
        snapshot = self._snapshot(source, event)
        if snapshot is not None:
            self._deliver(snapshot)

    def attach(self) -> None:
        assert _sinks is not None
        self._token = _sinks.add_stream_sink(self._sink)

    def detach(self) -> None:
        assert _sinks is not None
        if self._token is not None:
            # A ValueError: exited in another context than it was entered in. Threads started in the block keep the
            # sink in their copies of the context anyway; the stream ignores what they emit once it is closed.
            with contextlib.suppress(ValueError):
                _sinks.reset_stream_sinks(self._token)
            self._token = None

    def settle(self) -> None:
        pass


# The fallback: one handler per event class, registered once per process, finds the stream through a ContextVar that
# the block sets. The bus copies the emitting context into the pool task of a synchronous handler, so the variable
# scopes runs as the sink does.
_active: contextvars.ContextVar[BusTransport | None] = contextvars.ContextVar("iq_stream_crewai_bus", default=None)
_registered = False
_register_lock = threading.Lock()
_STOP = object()


def _event_classes(root: type[BaseEvent] = BaseEvent) -> Iterator[type[BaseEvent]]:
    for kind in root.__subclasses__():
        yield kind
        yield from _event_classes(kind)


def _handler(source: object, event: BaseEvent) -> None:
    transport = _active.get()
    if transport is not None:
        transport.take(source, event)


def _register() -> None:
    global _registered
    from crewai.events import crewai_event_bus

    with _register_lock:
        if _registered:
            return
        for kind in set(_event_classes()):
            default = kind.model_fields.get("type")
            if default is not None and default.default in EVENT_FIELDS:
                crewai_event_bus.register_handler(kind, _handler)
        _registered = True


class BusTransport:
    """Handlers on the global event bus. They run in the bus's thread pool, after any other listener of the same
    event and out of emission order, so a writer thread holds each event for ``window`` seconds and releases what it
    holds in ``emission_sequence`` order; on exit the bus is flushed first. Degraded: it adds latency, depends on a
    timing margin, and ``flush()`` waits for every listener in the process."""

    blocking = True

    def __init__(self, snapshot: Snapshot, deliver: Deliver, window: float = 0.05, flush_timeout: float = 30.0) -> None:
        self._snapshot = snapshot
        self._deliver = deliver
        self._window = window
        self._flush_timeout = flush_timeout
        self._queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._token: contextvars.Token[BusTransport | None] | None = None
        self._writer = threading.Thread(target=self._write, name="iq-stream-crewai bus", daemon=True)

    def take(self, source: object, event: BaseEvent) -> None:
        snapshot = self._snapshot(source, event)
        if snapshot is not None:
            self._queue.put((event.emission_sequence or 0, time.monotonic(), snapshot))

    def attach(self) -> None:
        _register()
        self._writer.start()
        self._token = _active.set(self)

    def detach(self) -> None:
        if self._token is not None:
            with contextlib.suppress(ValueError):
                _active.reset(self._token)
            self._token = None

    def settle(self) -> None:
        from crewai.events import crewai_event_bus

        # Handlers of the block's last events may still be pending in the pool; they run with the block's context.
        crewai_event_bus.flush(timeout=self._flush_timeout)
        self._queue.put(_STOP)
        self._writer.join(self._flush_timeout)

    def _write(self) -> None:
        held: list[tuple[int, int, float, object]] = []
        count = 0
        stopping = False
        while not stopping or held:
            try:
                item = self._queue.get(timeout=self._window / 2 if held else None)
            except queue.Empty:
                item = None
            if item is _STOP:
                stopping = True
            elif isinstance(item, tuple):
                seq, arrived, snapshot = item  # pyright: ignore[reportUnknownVariableType]
                count += 1
                heapq.heappush(held, (seq, count, arrived, snapshot))  # pyright: ignore[reportUnknownArgumentType]
            now = time.monotonic()
            while held and (stopping or now - held[0][2] >= self._window):
                self._deliver(heapq.heappop(held)[3])
