"""The writer's state machine, shared by the two publishers.

It is written without I/O. ``Writer.run()`` is a program that yields what it needs done (open the held request,
send a short request, pause) and is sent back the answers; ``Writer.body()`` yields what to write into the held
request, or how long to wait. The thread of ``Publisher`` and the task of ``AsyncPublisher`` carry out the same
programs, each with its own I/O, so the rules of core-client.md live here once.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, replace
from time import monotonic
from typing import TypeVar

from ._answer import (
    NO_ANSWER,
    Answer,
    PublishResult,
    is_closed,
    is_retryable,
    last_event_id,
    refusal,
    to_result,
    usable,
)
from ._errors import IQError
from ._frame import NEWLINE, checked_frame, close_line
from ._protocol import Outcome

logger = logging.getLogger("iq_stream")

#: The time a short request (a probe, a close line, a reservation) may take.
POST_TIMEOUT = 10.0
#: How long giving up may take: the time allowed to tell subscribers at once that the stream failed.
OUT_OF_BAND = 3.0
#: A request that lived this long was a working one, and its loss starts a new series of retries.
HEALTHY = 1.0


def backoff(tries: int) -> float:
    return min(0.1 * 2**tries, 2.0)


@dataclass(frozen=True)
class Settings:
    """The client's options, resolved."""

    keepalive: float | None = 20.0
    min_open: float = 3.0
    max_frame_bytes: int = 65_536
    raise_errors: bool = False
    error_handler: Callable[[IQError], object] | None = None
    replay_frames: int = 512
    replay_bytes: int = 4 << 20
    queue_bytes: int = 8 << 20
    retry_budget: float = 10.0
    takeover_budget: float = 65.0


@dataclass(frozen=True)
class Stream:
    """Open the held request, with ``Writer.body()`` as its body, and send back its answer."""


@dataclass(frozen=True)
class Post:
    """Send a short request with this body and send back its answer."""

    body: bytes
    timeout: float


@dataclass(frozen=True)
class Pause:
    """Wait. An interruptible pause ends early when the publisher stops, and on nothing else."""

    seconds: float
    interruptible: bool = True


Command = Stream | Post | Pause
Program = Generator[Command, Answer, None]

STREAM = Stream()


@dataclass(frozen=True)
class Write:
    """Write these bytes into the held request. The body resumes once they were handed to the connection."""

    data: bytes


@dataclass(frozen=True)
class Idle:
    """Nothing to write: wait this long, or until something is sent, closed or stopped."""

    seconds: float


BodyCommand = Write | Idle


def post_within(body: bytes, retry: float, within: float = math.inf) -> Generator[Command, Answer, Answer]:
    """A short request, retried while the server says "not now" and ``retry`` lasts. ``retry`` bounds the retries
    only; ``within`` bounds the whole of it, the requests included."""
    started = monotonic()
    deadline = started + within
    retry_until = min(started + retry, deadline)
    tries = 0
    while True:
        answer = yield Post(body, max(0.001, min(POST_TIMEOUT, deadline - monotonic())))
        if not is_retryable(answer) or monotonic() + backoff(tries) >= retry_until:
            return answer
        yield Pause(backoff(tries), interruptible=False)
        tries += 1


def close_stream_within(outcome: Outcome, retry: float, within: float = math.inf) -> Generator[Command, Answer, Answer]:
    """A short request holding only a close line."""
    return post_within(close_line(outcome), retry, within)


T = TypeVar("T")


def drive(program: Generator[Command, Answer, T], carry_out: Callable[[Command], Answer]) -> T:
    """Carries out a program's commands, one at a time, with blocking I/O."""
    try:
        command = next(program)
        while True:
            command = program.send(carry_out(command))
    except StopIteration as stop:
        return stop.value
    finally:
        program.close()


async def adrive(program: Generator[Command, Answer, T], carry_out: Callable[[Command], Awaitable[Answer]]) -> T:
    """The same, on an event loop."""
    try:
        command = next(program)
        while True:
            command = program.send(await carry_out(command))
    except StopIteration as stop:
        return stop.value
    finally:
        program.close()


@dataclass
class _GiveUp:
    result: PublishResult
    close_it: bool
    outcome: Outcome = "fail"


class Writer:
    """One stream's state: the queue that ``send()`` fills, the replay ring, the close, and what was given up.

    ``send()`` may be called from any thread; everything it shares with the writer is guarded by one lock. The
    three hooks are the driver's: ``wake`` ends an ``Idle``, ``stopping`` drops the connection in use and ends an
    interruptible pause, ``settled`` says that the result is known.
    """

    def __init__(
        self,
        settings: Settings,
        stream_id: str,
        *,
        wake: Callable[[], None],
        stopping: Callable[[], None],
        settled: Callable[[], None],
    ) -> None:
        self.settings = settings
        self.stream_id = stream_id
        self._wake = wake
        self._stopping = stopping
        self._settled = settled
        self._lock = threading.Lock()
        self._created = monotonic()

        # Lines not yet handed to a request, and the bytes handed to one whose write has not completed.
        self._queue: deque[bytes] = deque()
        self._queued_bytes = 0
        self._in_flight = 0
        # The most recent data frames handed to a request: those numbered written - len(ring) + 1 to written.
        self._ring: deque[bytes] = deque()
        self._ring_bytes = 0
        self._written = 0
        # Frames the server does not hold, to send first on the next request.
        self._resend: list[bytes] = []

        self.frames_sent = 0
        self.bytes_sent = 0
        self.requests = 0
        self._closing: Outcome | None = None
        self._close_at = 0.0
        self._close_sent = False
        self._stopped = False
        self._give_up: _GiveUp | None = None
        self.error: IQError | None = None
        # The error is reported once the result is known, when subscribers have heard of the failure: a handler that
        # takes its time, or calls close(), then holds nothing up.
        self._unreported: IQError | None = None
        self.result: PublishResult | None = None

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def alive(self) -> bool:
        return not self._stopped and self._closing is None

    def send(self, data: str, event: str | None) -> bool:
        line = checked_frame(data, event, self.settings.max_frame_bytes)
        full: IQError | None = None
        with self._lock:
            if self.error is not None and self.settings.raise_errors:
                raise self.error.with_traceback(None)
            if self._stopped or self._closing is not None:
                return False
            if self._queued_bytes + self._in_flight + len(line) > self.settings.queue_bytes:
                # The server stopped reading: a stalled network must not hold the agent's memory.
                full = refusal(self.stream_id, "the queue is full, the server is not reading", NO_ANSWER, "queue_full")
            else:
                self._queue.append(line)
                self._queued_bytes += len(line)
                self.frames_sent += 1
                self.bytes_sent += len(line)
        if full is not None:
            self.give_up(full, None, close_it=True)
            if self.settings.raise_errors:
                raise full
            return False
        self._wake()
        return True

    def request_close(self, outcome: Outcome) -> float:
        """Queues the close line, held back until the publisher is ``min_open`` old. Returns when it may go out."""
        with self._lock:
            if not self._stopped and self._closing is None:
                self._closing = outcome
                self._close_at = max(monotonic(), self._created + self.settings.min_open)
            close_at = self._close_at if self._closing is not None else monotonic()
        self._wake()
        return close_at

    def abort(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stopping()
        self._settle(PublishResult(status=0, closed=False))

    def give_up(self, error: IQError, answer: Answer | None, close_it: bool) -> None:
        """Fail open: the error is recorded, the publisher goes inert, and the run goes on. With ``close_it`` the
        server may still hold the stream open, and ``finish()`` tells it that it failed, so that subscribers get
        ``iq:fail`` now and not after the idle timeout. The error is reported once, with the result."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self.error = self._unreported = error
            result = to_result(answer) if answer is not None and answer.status != 0 else PublishResult(status=0)
            if error.code is not None:
                result = replace(result, error=error.code)
            self._give_up = _GiveUp(result, close_it)
        self._stopping()

    def cancelled(self) -> None:
        """The writer's task was cancelled, as at the shutdown of its event loop: the stream is closed out of band,
        with the outcome already asked for (``end`` only if the server holds every frame), else with ``fail``."""
        with self._lock:
            if self._stopped:
                return
            outcome = self._closing
            if outcome is not None:
                self._stopped = True
                self._give_up = _GiveUp(PublishResult(status=0), True, outcome)
        if outcome is not None:
            self._stopping()
            return
        error = IQError(
            f"IQ stream {self.stream_id}: the publisher's task was cancelled before the stream was closed (cancelled)",
            stream_id=self.stream_id,
            code="cancelled",
        )
        self.give_up(error, None, close_it=True)

    def settled_result(self) -> PublishResult:
        """The result, once known; settles one from what went wrong when the writer could not."""
        if self.result is None:
            self._settle(PublishResult(status=0, error=None if self.error is None else self.error.code))
        assert self.result is not None
        return self.result

    def run(self) -> Program:
        """The writer: holds a request open, and acts on every answer the server can give."""
        failing_since: float | None = None
        tries = 0
        while not self._stopped:
            started = monotonic()
            self.requests += 1
            answer = usable((yield STREAM))
            with self._lock:
                self._in_flight = 0
            if not is_retryable(answer) or monotonic() - started > HEALTHY:
                failing_since = None
                tries = 0
            # Until an answer says what the stream holds, ask with an empty POST: the server answers it with the
            # stream's state. After a lost connection the first one goes out at once.
            while not self._stopped and is_retryable(answer):
                if failing_since is None:
                    failing_since = monotonic()
                if answer.status != 0 or tries > 0:
                    # The server counts a dead request of ours as attached until its read timeout passes, which is
                    # longer than an outage is worth waiting for: 409 has its own budget.
                    budget = self.settings.takeover_budget if answer.status == 409 else self.settings.retry_budget
                    if monotonic() + backoff(tries) > failing_since + budget:
                        code = "unreachable" if answer.status == 0 else None
                        self.give_up(refusal(self.stream_id, "gave up retrying", answer, code), answer, close_it=True)
                        return
                    yield Pause(backoff(tries))
                    if self._stopped:
                        return
                tries += 1
                answer = usable((yield Post(b"", POST_TIMEOUT)))
            if self._stopped:
                return

            if is_closed(answer.body):
                if answer.status == 200 or self._closed_here(answer):
                    with self._lock:
                        self._stopped = True
                    self._settle(to_result(answer))
                    return
                self.give_up(refusal(self.stream_id, "the stream is closed", answer), answer, close_it=False)
                return
            held = last_event_id(answer.body)
            if answer.status in (200, 408) and held is not None:
                if self._resume(held):
                    continue
                # A stream with a hole in it is worse for a client than one that says it failed. This is also how a
                # restarted server shows: the tail of an answer is not published into a fresh stream.
                what = (
                    f"the server holds {held} frames, {self._written} were written, "
                    "and the replay buffer cannot bridge them"
                )
                self.give_up(refusal(self.stream_id, what, answer, "gap"), answer, close_it=True)
                return
            # A refused line leaves the stream open. Bad credentials or an app at its limit: there is nothing to close.
            self.give_up(
                refusal(self.stream_id, "the server refused the request", answer), answer, answer.status in (400, 413)
            )
            return

    def finish(self) -> Program:
        """What giving up left to do: the close line out of band, then the result."""
        with self._lock:
            pending, self._give_up = self._give_up, None
        if pending is None:
            self.settled_result()
            return
        result = pending.result
        if pending.close_it:
            deadline = monotonic() + OUT_OF_BAND
            outcome = pending.outcome
            closed: Answer | None = None
            if outcome == "end":
                # Only a cancellation leaves `end` to send here, and it may have cut the request short: the server
                # alone knows whether it holds every frame, and an answer with a hole in it ends with `fail`.
                probe = yield from post_within(b"", self.settings.retry_budget, OUT_OF_BAND)
                if is_closed(probe.body):
                    closed = probe
                elif last_event_id(probe.body) != self.frames_sent:
                    outcome = "fail"
                    error = IQError(
                        f"IQ stream {self.stream_id}: the publisher's task was cancelled before every frame reached "
                        "the server (cancelled)",
                        stream_id=self.stream_id,
                        code="cancelled",
                    )
                    with self._lock:
                        self.error = self._unreported = error
                    result = replace(result, error=error.code)
            if closed is None:
                closed = yield from close_stream_within(
                    outcome, self.settings.retry_budget, max(0.0, deadline - monotonic())
                )
            if is_closed(closed.body):
                # The result keeps the status and the code of what went wrong, and says how the stream ended.
                said = to_result(closed)
                result = replace(
                    result,
                    closed=True,
                    last_event_id=said.last_event_id if said.last_event_id is not None else result.last_event_id,
                    outcome=said.outcome or result.outcome,
                    reason=said.reason or result.reason,
                )
                if self.error is None and result.status == 0:
                    result = replace(said, error=None if said.error == "stream_closed" else said.error)
        self._settle(result)

    def body(self, answered: Callable[[], bool]) -> Generator[BodyCommand, None, None]:
        """One held request's body: pending resends, then the queue as it fills, keepalives, the close line.

        ``answered`` says whether the server has answered the request or dropped it. It is asked before anything is
        taken from the queue: the server reads nothing more, and what is still queued goes into the next request."""
        keepalive = math.inf if self.settings.keepalive is None else self.settings.keepalive
        if answered():
            return
        with self._lock:
            resend, self._resend = self._resend, []
            self._in_flight = sum(len(line) for line in resend)
        last_write = monotonic()
        if resend:
            yield Write(b"".join(resend))
            last_write = self._wrote()
        while True:
            final = False
            wait = 0.0
            if answered():
                return
            with self._lock:
                if self._stopped:
                    return
                now = monotonic()
                data: bytes | None
                if self._queue:
                    data = self._take()
                elif self._closing is not None and now >= self._close_at:
                    self._close_sent = True
                    data = close_line(self._closing)
                    final = True
                elif now - last_write >= keepalive:
                    data = NEWLINE
                else:
                    data = None
                    wait = min(last_write + keepalive, self._close_at if self._closing is not None else math.inf) - now
            if data is None:
                yield Idle(wait)
                continue
            yield Write(data)
            last_write = self._wrote()
            if final:
                return

    def _wrote(self) -> float:
        with self._lock:
            self._in_flight = 0
        return monotonic()

    def _take(self) -> bytes:
        """Moves the queue into the replay ring and returns it as one chunk: lines queued together leave together."""
        lines = list(self._queue)
        self._queue.clear()
        self._queued_bytes = 0
        self._written += len(lines)
        for line in lines:
            self._ring.append(line)
            self._ring_bytes += len(line)
        while len(self._ring) > self.settings.replay_frames or self._ring_bytes > self.settings.replay_bytes:
            self._ring_bytes -= len(self._ring.popleft())
        data = b"".join(lines)
        self._in_flight = len(data)
        return data

    def _resume(self, held: int) -> bool:
        """Puts the frames the server does not hold at the head of the next request. False when the ring cannot
        bridge the gap."""
        with self._lock:
            oldest = self._written - len(self._ring) + 1
            if held > self._written or held + 1 < oldest:
                return False
            self._resend = list(self._ring)[held + 1 - oldest :]
            return True

    def _closed_here(self, answer: Answer) -> bool:
        """A ``409 stream_closed`` that reports the close line this publisher sent, with its outcome: the line
        arrived, its answer was lost, and the probe that followed heard of it."""
        body = answer.body
        return (
            answer.status == 409
            and body.get("error") == "stream_closed"
            and self._close_sent
            and body.get("outcome") == self._closing
            and body.get("reason") in (None, "publisher")
        )

    def _settle(self, result: PublishResult) -> None:
        """Records the result, reports the error, then says that the result is known: whoever waits in ``close()``
        returns with the error reported."""
        with self._lock:
            if self.result is not None:
                return
            self.result = result
            error, self._unreported = self._unreported, None
        if error is not None:
            self._report(error)
        self._settled()

    def _report(self, error: IQError) -> None:
        try:
            if self.settings.error_handler is not None:
                self.settings.error_handler(error)
            elif not self.settings.raise_errors:
                logger.warning("%s", error)
        except Exception:
            # A failing handler must not break the run either.
            pass
