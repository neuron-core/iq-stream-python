"""What every encoder shares: the frame type, the ``Encoder`` protocol, and the helpers that keep a frame JSON, small
and named as the server accepts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from workflows.events import Event

#: One frame: the SSE event name (``None`` for an unnamed frame) and the data string.
Frame = tuple["str | None", str]

#: Whether a frame fits the publisher's line limit, envelope and escapes included.
Fits = Callable[[Frame], bool]

#: What a browser is told instead of an exception's text, unless the encoder is built with ``expose_errors=True``.
MASKED_ERROR = "An error occurred."

#: The default bound of one large field (tool arguments, a tool result, a final answer), in UTF-8 bytes.
DEFAULT_MAX_FIELD_BYTES = 16_384

_EVENT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_SURROGATE = re.compile("[\ud800-\udfff]")


class Encoder(Protocol):
    """Turns a run's events into frames. ``publish()`` calls it from one task, in the order of the stream. An encoder
    serves one run at a time: ``start()`` begins a run, so one instance may serve runs one after another, not two at
    once."""

    def start(self, run_id: str) -> Iterable[Frame]:
        """The frames that open the stream, before the first event. It begins a new run: the state an encoder keeps
        about the previous one is dropped here."""
        ...

    def encode(self, event: Event) -> Iterable[Frame]:
        """The frames of one stream event, the terminal ``StopEvent`` (or subclass) included. Most events give one
        frame, some none, some several; the frames of one event are queued together."""
        ...

    def abort(self, error: BaseException) -> Iterable[Frame]:
        """The frames that end a stream whose run ended without a terminal event, or whose answer could not be
        saved: ``error`` is why, an ``asyncio.CancelledError`` for a run that was stopped."""
        ...

    def fit(self, frame: Frame, fits: Fits) -> Iterable[Frame]:
        """What to send instead of ``frame``, which is over the publisher's line limit: a shorter frame, several
        frames that each fit, or nothing to drop it. ``fits`` tells whether a frame fits."""
        ...


def dumps(value: object) -> str:
    """Compact JSON, non-ASCII kept, anything JSON has no type for as its ``str()``, a non-finite float as ``null``
    and a lone surrogate as U+FFFD."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if "NaN" in text or "Infinity" in text:
        # Python writes non-finite floats as bare words that JSON.parse refuses; strings are left as they are.
        plain = json.loads(text, parse_constant=lambda _: None)
        text = json.dumps(plain, ensure_ascii=False, separators=(",", ":"))
    return scrub(text)


def scrub(text: str) -> str:
    """``text`` with each lone surrogate, which has no UTF-8 form, as U+FFFD, as the server would read it. They come
    from a surrogate pair split across two streamed chunks, or from text decoded with ``surrogateescape``."""
    if text.isascii():
        return text
    return _SURROGATE.sub("�", text)


def call_key(tool_id: str, tool_name: str, kwargs: dict[str, Any]) -> tuple[str, str, str]:
    """What tells a tool call apart from another with the same id: some providers have no ids and use the tool's name
    (Ollama, sometimes Gemini), so a model that calls one tool twice gives two calls with one id."""
    return tool_id, tool_name, json.dumps(kwargs, sort_keys=True, default=str)


def event_name(event: object) -> str:
    """The event's class name when the server accepts it as an SSE event name, else ``Event``."""
    name = type(event).__name__
    return name if _EVENT_NAME.fullmatch(name) else "Event"


def fields(event: Event) -> dict[str, Any]:
    """An event's fields as JSON values: the declared ones and the undeclared ones, which workflows keep apart under
    ``_data`` and are merged here into the top level."""
    data: dict[str, Any] = event.model_dump(mode="json", by_alias=True, fallback=str)
    extra = data.pop("_data", None)
    if isinstance(extra, dict):
        data.update(extra)  # pyright: ignore[reportUnknownArgumentType]
    return data


def byte_size(text: str) -> int:
    return len(text.encode())


def cut(text: str, max_bytes: int) -> str:
    """The longest prefix of ``text`` whose UTF-8 encoding is at most ``max_bytes``."""
    encoded = text.encode()
    if len(encoded) <= max_bytes:
        return text
    return encoded[: max(max_bytes, 0)].decode("utf-8", errors="ignore")


def cap_strings(value: Any, cap: int, marker: str, keep: frozenset[str] = frozenset()) -> Any:
    """``value`` with every string longer than ``cap`` UTF-8 bytes cut to ``cap`` and suffixed with ``marker``. The
    values of the keys in ``keep`` (ids, names, types) are left alone."""
    if isinstance(value, str):
        return cut(value, cap) + marker if byte_size(value) > cap else value
    if isinstance(value, dict):
        return {
            key: item if key in keep else cap_strings(item, cap, marker, keep)
            for key, item in value.items()  # pyright: ignore[reportUnknownVariableType]
        }
    if isinstance(value, list):
        return [cap_strings(item, cap, marker, keep) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def longest_string(value: Any, keep: frozenset[str] = frozenset()) -> int:
    """The UTF-8 size of the longest string in ``value``, outside the keys in ``keep``."""
    if isinstance(value, str):
        return byte_size(value)
    if isinstance(value, dict):
        sizes = [
            longest_string(item, keep)
            for key, item in value.items()  # pyright: ignore[reportUnknownVariableType]
            if key not in keep
        ]
        return max(sizes, default=0)
    if isinstance(value, list):
        return max((longest_string(item, keep) for item in value), default=0)  # pyright: ignore[reportUnknownVariableType]
    return 0


def largest(fits_at: Callable[[int], bool], high: int) -> int:
    """The largest ``n`` in ``0..high`` for which ``fits_at(n)`` holds, or -1; ``fits_at`` must hold for every ``n``
    below one for which it holds."""
    low, best, probe = 0, -1, 1
    # Grow a bound first, so that cutting a long text into small frames builds frames near the size that fits.
    while probe <= high and fits_at(probe):
        best, low, probe = probe, probe + 1, probe * 2
    high = min(high, probe - 1)
    while low <= high:
        middle = (low + high) // 2
        if fits_at(middle):
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


def shrink_strings(
    value: Any, build: Callable[[Any], Frame], fits: Fits, marker: str, keep: frozenset[str] = frozenset()
) -> Frame | None:
    """The frame ``build`` makes of ``value`` with its strings cut as little as it takes to fit, or ``None``."""

    def capped(cap: int) -> Frame:
        return build(cap_strings(value, cap, marker, keep))

    cap = largest(lambda n: fits(capped(n)), longest_string(value, keep) - 1)
    return None if cap < 0 else capped(cap)


def bounded_json(value: Any, max_bytes: int, marker: str) -> tuple[Any, bool]:
    """``value`` as JSON values with its strings cut so that its JSON is at most ``max_bytes``, and whether anything
    was cut. A value that cannot be brought under the bound that way (a long list of numbers) becomes ``{}``."""
    plain = json.loads(dumps(value))
    if byte_size(dumps(plain)) <= max_bytes:
        return plain, False
    frame = shrink_strings(plain, lambda v: (None, dumps(v)), lambda f: byte_size(f[1]) <= max_bytes, marker)
    return (json.loads(frame[1]) if frame is not None else {}), True


def split_text(text: str, build: Callable[[str], Frame], fits: Fits) -> list[Frame] | None:
    """``text`` cut into consecutive pieces whose frames each fit, or ``None`` when one character does not fit."""
    frames: list[Frame] = []
    rest = text
    while rest:
        size = largest(_prefix_fits(rest, build, fits), len(rest))
        if size < 1:
            return None
        frames.append(build(rest[:size]))
        rest = rest[size:]
    return frames


def _prefix_fits(text: str, build: Callable[[str], Frame], fits: Fits) -> Callable[[int], bool]:
    return lambda size: fits(build(text[:size]))
