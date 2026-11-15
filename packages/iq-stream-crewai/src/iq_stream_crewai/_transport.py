"""How events reach a stream: a scoped stream sink, or handlers on CrewAI's global event bus as a fallback."""

from __future__ import annotations

import contextlib
import contextvars
import heapq
import queue
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, Literal, Protocol

from crewai.events.base_events import BaseEvent

from ._snapshot import EVENT_FIELDS, SCOPE_TYPES

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
        self._snapshot: Snapshot | None = snapshot
        self._deliver: Deliver | None = deliver
        self._token: contextvars.Token[Any] | None = None

    def _sink(self, source: object, event: BaseEvent) -> None:
        take, deliver = self._snapshot, self._deliver
        if take is None or deliver is None:
            return
        snapshot = take(source, event)
        if snapshot is not None:
            deliver(snapshot)

    def attach(self) -> None:
        assert _sinks is not None
        self._token = _sinks.add_stream_sink(self._sink)

    def detach(self) -> None:
        assert _sinks is not None
        if self._token is not None:
            # A ValueError: exited in another context than it was entered in.
            with contextlib.suppress(ValueError):
                _sinks.reset_stream_sinks(self._token)
            self._token = None
        # Copies of the block's context outlive it: threads started in the block, and the event loop CrewAI's bus
        # creates on its first event, in that event's context. The sink they keep lets go of the stream.
        self._snapshot = self._deliver = None

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
        # The scope events too, published or not: they tell where an event was emitted.
        wanted = EVENT_FIELDS.keys() | SCOPE_TYPES
        for kind in set(_event_classes()):
            default = kind.model_fields.get("type")
            if default is not None and default.default in wanted:
                crewai_event_bus.register_handler(kind, _handler)
        _registered = True


def _counter_variable() -> contextvars.ContextVar[object] | None:
    """CrewAI's emission counter: a new one at every top-level kickoff, shared by the threads and tasks of that
    kickoff, the iterator ``emission_sequence`` comes from."""
    from crewai.events import base_events

    variable: object = getattr(base_events, "_emission_counter", None)
    return variable if isinstance(variable, contextvars.ContextVar) else None  # pyright: ignore[reportUnknownVariableType]


class BusTransport:
    """Handlers on the global event bus. They run in the bus's thread pool, after any other listener of the same
    event and out of emission order, so a writer thread holds each event for ``window`` seconds and releases what it
    holds in the order the events were emitted; on exit the bus is flushed first. Degraded: it adds latency, depends
    on a timing margin, and ``flush()`` waits for every listener in the process."""

    blocking = True

    def __init__(self, snapshot: Snapshot, deliver: Deliver, window: float = 0.05, flush_timeout: float = 30.0) -> None:
        self._snapshot: Snapshot | None = snapshot
        self._window = window
        self._flush_timeout = flush_timeout
        self._queue: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._token: contextvars.Token[BusTransport | None] | None = None
        self._writer = threading.Thread(target=self._write, args=(deliver,), name="iq-stream-crewai bus", daemon=True)
        self._counter = _counter_variable()
        # The kickoffs seen, by their emission counter: when the first of their events was made, and in which order
        # they were seen.
        self._kickoffs: dict[object, tuple[float, int]] = {}
        self._kickoffs_lock = threading.Lock()

    def take(self, source: object, event: BaseEvent) -> None:
        take = self._snapshot
        if take is None:
            return  # settled: nothing more is delivered
        snapshot = take(source, event)
        if snapshot is not None:
            self._queue.put((self._order(event), time.monotonic(), snapshot))

    def _order(self, event: BaseEvent) -> tuple[float, int, int]:
        """Where an event goes among the others. ``emission_sequence`` orders the events of one kickoff, whatever the
        clock does; it restarts at every top-level kickoff, so the kickoffs are ordered by the time their first event
        seen here was made, then by the order they were seen. Runs in a copy of the emitting context."""
        sequence = event.emission_sequence or 0
        stamp: object = getattr(event, "timestamp", None)
        made = stamp.timestamp() if isinstance(stamp, datetime) else time.time()
        counter = None if self._counter is None else self._counter.get(None)
        if counter is None:
            return made, 0, sequence  # a CrewAI without the counter: the clock
        with self._kickoffs_lock:
            kickoff = self._kickoffs.get(counter)
            if kickoff is None:
                kickoff = self._kickoffs[counter] = (made, len(self._kickoffs))
        return (*kickoff, sequence)

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
        # Copies of the block's context outlive it, the one CrewAI's event loop was created in among them, and keep
        # this transport in _active: it lets go of the stream (the writer, which delivers, has ended).
        self._snapshot = None
        with self._kickoffs_lock:
            self._kickoffs.clear()

    def _write(self, deliver: Deliver) -> None:
        held: list[tuple[tuple[float, int, int], int, float, object]] = []
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
                order, arrived, snapshot = item  # pyright: ignore[reportUnknownVariableType]
                count += 1
                heapq.heappush(held, (order, count, arrived, snapshot))  # pyright: ignore[reportUnknownArgumentType]
            now = time.monotonic()
            while held and (stopping or now - held[0][2] >= self._window):
                deliver(heapq.heappop(held)[3])
