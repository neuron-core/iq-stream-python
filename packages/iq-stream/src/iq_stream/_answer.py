"""The server's answers to publish requests, and what they mean for a publisher."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from ._errors import IQError
from ._protocol import Outcome


@dataclass(frozen=True)
class Answer:
    """The server's answer to one request. ``status`` is 0 when the connection gave none."""

    status: int
    body: Mapping[str, object] = field(default_factory=lambda: {})


NO_ANSWER = Answer(0)


def parse_body(content: bytes) -> dict[str, object]:
    try:
        value: object = json.loads(content)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True, eq=False)
class PublishResult:
    """The server's answer to a publish request, or what the publisher knows when it had none."""

    #: HTTP status; 0 when the publisher gave up without an answer.
    status: int
    #: How many data frames the stream holds.
    last_event_id: int | None = None
    closed: bool = False
    outcome: Outcome | None = None
    #: Why the stream failed, when ``outcome`` is ``fail``.
    reason: str | None = None
    #: The server's error code, or the publisher's own, when the request was refused or the publisher gave up.
    error: str | None = None

    def _fields(self) -> tuple[object, ...]:
        return (self.status, self.last_event_id, self.closed, self.outcome, self.reason, self.error)

    # Results compare by value, whatever subclass carries them (the testing kit's is awaitable).
    def __eq__(self, other: object) -> bool:
        return isinstance(other, PublishResult) and self._fields() == other._fields()

    def __hash__(self) -> int:
        return hash(self._fields())


def last_event_id(body: Mapping[str, object]) -> int | None:
    value = body.get("lastEventId")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def is_closed(body: Mapping[str, object]) -> bool:
    return body.get("closed") is True or body.get("error") == "stream_closed"


def is_retryable(answer: Answer) -> bool:
    """No answer, or one that says "not now": the stream may still be there, so ask again."""
    return answer.status in (0, 502, 503, 504) or (
        answer.status == 409 and answer.body.get("error") == "publisher_active"
    )


def usable(answer: Answer) -> Answer:
    """A 200 or 408 that does not say how many frames the stream holds cannot be acted on: no answer."""
    if answer.status in (200, 408) and not is_closed(answer.body) and last_event_id(answer.body) is None:
        return NO_ANSWER
    return answer


def to_result(answer: Answer) -> PublishResult:
    body = answer.body
    outcome = body.get("outcome")
    reason = body.get("reason")
    error = body.get("error")
    return PublishResult(
        status=answer.status,
        last_event_id=last_event_id(body),
        closed=is_closed(body),
        outcome="end" if outcome == "end" else "fail" if outcome == "fail" else None,
        reason=reason if isinstance(reason, str) else None,
        error=error if isinstance(error, str) else None,
    )


def refusal(stream_id: str, what: str, answer: Answer = NO_ANSWER, code: str | None = None) -> IQError:
    """An error with the stream id, the status and the server's code, and never a token or frame data."""
    server_code = answer.body.get("error")
    error = code if code is not None else server_code if isinstance(server_code, str) else None
    if answer.status == 0:
        detail = error or "no answer"
    else:
        detail = f"HTTP {answer.status}" + ("" if error is None else f", {error}")
    return IQError(
        f"IQ stream {stream_id}: {what} ({detail})",
        stream_id=stream_id,
        status=answer.status or None,
        code=error,
    )
