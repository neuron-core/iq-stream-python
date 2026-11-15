"""Frames that fit the publisher's ``max_frame_bytes``: a delta is split over several frames, a value that cannot be
split is cut with a marker, and a frame that still does not fit is dropped and reported."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from iq_stream import FrameTooLarge

Frame = tuple["str | None", str]
"""One frame: the SSE event name (``None`` for an unnamed frame) and the data string."""

# The longest a character of a member value gets on the wire: escaped once in the frame's JSON and again in the
# line's envelope (a control character becomes ``\u0001`` and then ``\\u0001``).
_WORST = 7
# ``{"event":"","data":""}``, with room to spare.
_ENVELOPE = 24
# Bytes on the wire of each ASCII character inside a member value.
_ASCII_COST = [7] * 32 + [1] * 96
for _escaped, _bytes in (("\b", 3), ("\f", 3), ("\n", 3), ("\r", 3), ("\t", 3), ('"', 4), ("\\", 4)):
    _ASCII_COST[ord(_escaped)] = _bytes


def dumps(value: object) -> str:
    """Compact JSON with non-ASCII text left as it is, as AG-UI's own encoder writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def marker(dropped: int) -> str:
    """What replaces the end of a value that was cut: how many bytes of UTF-8 were dropped."""
    return f"… [{dropped} bytes truncated]"


def utf8_size(text: str) -> int:
    # surrogatepass: a lone surrogate, which JSON text can carry, counts for the three bytes it is written with.
    return len(text.encode("utf-8", "surrogatepass"))


def cut_text(text: str, max_bytes: int | None) -> str:
    """``text`` cut to ``max_bytes`` bytes of UTF-8, at a character's boundary, plus a marker that says how many bytes
    were dropped."""
    if max_bytes is None or 4 * len(text) <= max_bytes:
        return text
    data = text.encode("utf-8", "surrogatepass")
    if len(data) <= max_bytes:
        return text
    kept = data[:max_bytes].decode("utf-8", "ignore")  # without a character cut in two
    return kept + marker(len(data) - utf8_size(kept))


def _cost(char: str) -> int:
    code = ord(char)
    if code < 128:
        return _ASCII_COST[code]
    if code < 0x800:
        return 2
    if 0xD800 <= code <= 0xDFFF:
        return 6  # a lone surrogate is escaped in the line
    return 3 if code < 0x10000 else 4


def _prefix(text: str, start: int, budget: int) -> int:
    """The end of the longest run of ``text`` from ``start`` whose characters fit ``budget`` bytes."""
    end = start
    used = 0
    length = len(text)
    while end < length:
        used += _cost(text[end])
        if used > budget:
            break
        end += 1
    return end


class Fitter:
    """Makes the frames of one stream fit its limit. Built-in encoders build every frame through it."""

    def __init__(
        self,
        limit: int,
        frame_size: Callable[[str, str | None], int],
        on_drop: Callable[[FrameTooLarge], None],
    ) -> None:
        self.limit = limit
        self._frame_size = frame_size
        self._on_drop = on_drop

    def fits(self, data: str, event: str | None = None) -> bool:
        # Every character of JSON text takes at most six bytes in the line: most frames need no measuring.
        if 6 * len(data) + len(event or "") + _ENVELOPE <= self.limit:
            return True
        return self._frame_size(data, event) <= self.limit

    def _drop(self, data: str, event: str | None) -> list[Frame]:
        self._on_drop(FrameTooLarge(self._frame_size(data, event), self.limit))
        return []

    def whole(self, payload: Mapping[str, Any], event: str | None = None) -> list[Frame]:
        """The payload as one frame, or nothing (reported) when it does not fit."""
        data = dumps(payload)
        return [(event, data)] if self.fits(data, event) else self._drop(data, event)

    def split(self, payload: Mapping[str, Any], key: str, event: str | None = None) -> list[Frame]:
        """The payload as one frame, or as several whose ``key`` members concatenate to the original string."""
        data = dumps(payload)
        if self.fits(data, event):
            return [(event, data)]
        text: str = payload[key]
        base = dict(payload)
        base[key] = ""
        budget = self.limit - self._frame_size(dumps(base), event)
        if budget < _WORST:
            return self._drop(data, event)
        frames: list[Frame] = []
        start = 0
        while start < len(text):
            end = max(_prefix(text, start, budget), start + 1)
            frames.extend(self._piece(base, key, text[start:end], event))
            start = end
        return frames

    def _piece(self, base: dict[str, Any], key: str, text: str, event: str | None) -> list[Frame]:
        base[key] = text
        data = dumps(base)
        if self.fits(data, event):
            return [(event, data)]
        if len(text) == 1:
            return self._drop(data, event)
        # The estimate is an upper bound, so this does not happen; halving keeps it correct if it ever did.
        middle = len(text) // 2
        return self._piece(base, key, text[:middle], event) + self._piece(base, key, text[middle:], event)

    def cut(
        self, payload: Mapping[str, Any], keys: Sequence[str], event: str | None = None, *, within: str | None = None
    ) -> list[Frame]:
        """The payload as one frame, its longest ``keys`` members cut with a marker until it fits. A member that is
        not a string is cut as its JSON text. With ``within``, the keys are members of that member of the payload."""
        data = dumps(payload)
        if self.fits(data, event):
            return [(event, data)]
        shrunk = dict(payload)
        target: dict[str, Any] = shrunk
        if within is not None:
            target = dict(payload[within])
            shrunk[within] = target
        present = [key for key in keys if target.get(key) is not None]
        texts = {key: target[key] if isinstance(target[key], str) else dumps(target[key]) for key in present}
        for key in sorted(present, key=lambda name: len(texts[name]), reverse=True):
            text = texts[key]
            size = utf8_size(text)
            target[key] = marker(size)  # the longest the marker can be
            budget = self.limit - self._frame_size(dumps(shrunk), event)
            end = _prefix(text, 0, budget) if budget > 0 else 0
            target[key] = text[:end] + marker(size - utf8_size(text[:end]))
            data = dumps(shrunk)
            if self.fits(data, event):
                return [(event, data)]
        return self._drop(data, event)
