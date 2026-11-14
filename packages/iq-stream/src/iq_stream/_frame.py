"""Frames as NDJSON lines."""

from __future__ import annotations

import json
import re

from ._errors import FrameTooLarge
from ._protocol import EVENT_NAME, Outcome

# Non-ASCII text and slashes stay unescaped, as JSON.stringify writes them: lines stay small and the server
# takes its fast path for data without escapes.
_ENCODER = json.JSONEncoder(ensure_ascii=False)
_SURROGATE = re.compile("[\ud800-\udfff]")

NEWLINE = b"\n"


def encode_frame(data: str, event: str | None = None) -> bytes:
    """One frame as an NDJSON line, newline included."""
    if not isinstance(data, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("Frame data must be a string.")
    if event is None:
        text = '{"data":' + _ENCODER.encode(data) + "}\n"
    elif isinstance(event, str) and EVENT_NAME.fullmatch(event):  # pyright: ignore[reportUnnecessaryIsInstance]
        text = '{"event":"' + event + '","data":' + _ENCODER.encode(data) + "}\n"
    else:
        raise ValueError("Invalid event name: 1 to 64 characters of A-Z a-z 0-9 _ . -")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate has no UTF-8 form. Escaped, as JSON.stringify writes it, the server reads it as U+FFFD.
        return _SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", text).encode("utf-8")


def frame_size(data: str, event: str | None = None) -> int:
    """The size the server measures against its limit: the line in UTF-8 bytes, without its newline."""
    return len(encode_frame(data, event)) - 1


def checked_frame(data: str, event: str | None, limit: int) -> bytes:
    line = encode_frame(data, event)
    if len(line) - 1 > limit:
        raise FrameTooLarge(len(line) - 1, limit)
    return line


def close_line(outcome: Outcome) -> bytes:
    return b'{"close":"' + outcome.encode() + b'"}\n'
