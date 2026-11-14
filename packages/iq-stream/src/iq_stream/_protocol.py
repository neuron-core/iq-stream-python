"""The parts of the IQ-Stream wire protocol that the client relies on (the server's docs/transport-protocol.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: Stream ids: 16 to 128 characters of the base64url alphabet.
STREAM_ID = re.compile(r"[A-Za-z0-9_-]{16,128}")

#: Event names of frames. A colon is impossible, which keeps the ``iq:`` names the server's own.
EVENT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")

#: The server's default limit on one NDJSON line, in bytes.
DEFAULT_MAX_FRAME_BYTES = 65_536

#: The furthest ``exp`` the server accepts by default, in seconds from now.
DEFAULT_TOKEN_TTL = 3600

#: The server's own SSE events. Each ends a subscription.
TERMINAL_EVENTS = ("iq:end", "iq:fail", "iq:gone")

Scope = Literal["pub", "sub"]
"""What a token allows: publishing (``pub``) or subscribing (``sub``)."""

Outcome = Literal["end", "fail"]
"""How a stream is closed: ``end`` when the answer finished, ``fail`` when it did not."""


def check_stream_id(stream_id: object) -> str:
    if not isinstance(stream_id, str) or not STREAM_ID.fullmatch(stream_id):
        raise ValueError("Invalid stream id: 16 to 128 characters of A-Z a-z 0-9 _ -.")
    return stream_id


def check_scope(scope: object) -> Scope:
    if scope == "pub":
        return "pub"
    if scope == "sub":
        return "sub"
    raise ValueError('The scope is "pub" or "sub".')


def check_outcome(outcome: object) -> Outcome:
    if outcome == "end":
        return "end"
    if outcome == "fail":
        return "fail"
    raise ValueError('The close outcome is "end" or "fail".')


def check_seconds(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"The {name} is a number of seconds.")
    seconds = float(value)
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0 or (positive and seconds == 0):
        raise ValueError(f"The {name} is a {'positive' if positive else 'non-negative'} number of seconds.")
    return seconds


@dataclass(frozen=True)
class Ticket:
    """What a browser needs to subscribe to a stream."""

    stream_id: str
    #: The ``sub`` token.
    token: str
    #: The subscribe URL with the token in its query: ready for ``new EventSource(url)``.
    url: str
    #: Unix seconds at which the token expires.
    expires_at: int

    def to_dict(self) -> dict[str, str | int]:
        """The ticket as the browser client of ``@neuron-core/iq-stream`` reads it: camelCase keys, ready for JSON."""
        return {"streamId": self.stream_id, "token": self.token, "url": self.url, "expiresAt": self.expires_at}
