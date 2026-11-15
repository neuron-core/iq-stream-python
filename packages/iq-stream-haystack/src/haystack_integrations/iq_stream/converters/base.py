"""What the converters share: the frame, the protocol a converter follows, the key of the caller, and the fitting
of text to the frame limit."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import uuid
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from haystack.dataclasses import StreamingChunk, TextContent, ToolCallResult
from iq_stream import DEFAULT_MAX_FRAME_BYTES, FrameTooLarge, frame_size

if TYPE_CHECKING:
    from typing_extensions import Unpack

#: The text a browser gets in place of an error message, unless the stream exposes errors.
MASKED_ERROR = "An error occurred."
#: The content of a tool result that is not published (``include_tool_results=False``).
OMITTED_RESULT = "[tool result omitted]"
#: Why a tool call whose arguments are not JSON gets no result: Haystack drops such a call.
INVALID_ARGUMENTS = "The tool call's arguments are not valid JSON."
#: The finish reasons that end one model message; ``tool_call_results`` ends a batch of tool results instead.
MODEL_FINISH = frozenset({"stop", "length", "content_filter", "tool_calls"})

_ENCODER = json.JSONEncoder(ensure_ascii=False)


@dataclass(frozen=True)
class Frame:
    """One frame of a stream: ``data`` (usually JSON) and an optional SSE event name."""

    data: str
    event: str | None = None


class ChunkConverter(Protocol):
    """Turns a run's ``StreamingChunk`` objects into frames. ``IQStream`` calls it under a lock, one call at a time,
    and sends every frame it returns, in order."""

    def start(self) -> Iterable[Frame]:
        """The frames that open the stream."""
        ...

    def convert(self, chunk: StreamingChunk, source: Hashable) -> Iterable[Frame]:
        """The frames of one chunk. ``source`` identifies the thread and task that called back: one model message
        always arrives from a single source, while sub-agents and tool threads may interleave."""
        ...

    def finish(self) -> Iterable[Frame]:
        """The frames of a run that completed."""
        ...

    def error(self, exc: BaseException) -> Iterable[Frame]:
        """The frames of a run that raised ``exc``."""
        ...

    def cancelled(self) -> Iterable[Frame]:
        """The frames of a run that was cancelled."""
        ...

    def custom(self, name: str, value: object) -> Iterable[Frame]:
        """A named event of the application's own, such as a request for a confirmation."""
        ...


def source_key() -> Hashable:
    """The caller: its thread and, in asyncio, its task."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), None if task is None else id(task)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def dumps(value: object) -> str:
    """Compact JSON, non-ASCII text unescaped, as the browser's JSON.parse reads it. What JSON cannot hold is written
    as its ``str()``, and a NaN or an infinity as ``null``, as JSON.stringify writes them: a bare ``NaN`` would make
    the browser reject the frame."""
    try:
        return _dumps(value)
    except ValueError:  # a non-finite float, or a reference cycle, which _finite raises again
        return _dumps(_finite(value, set()))


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False)


def _finite(value: object, seen: set[int]) -> object:
    """``value`` with every non-finite float in its dicts, lists and tuples replaced by ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if not isinstance(value, dict | list | tuple):
        return value
    marker = id(cast(object, value))
    if marker in seen:
        raise ValueError("Circular reference detected")
    seen.add(marker)
    if isinstance(value, dict):
        copy: object = {key: _finite(item, seen) for key, item in cast(dict[object, object], value).items()}
    else:
        copy = [_finite(item, seen) for item in cast(list[object] | tuple[object, ...], value)]
    seen.discard(marker)
    return copy


def result_text(result: ToolCallResult) -> str:
    """A tool result as the browser's text: the string, or the text parts with a placeholder for each image or file."""
    value = result.result
    if isinstance(value, str):
        return value
    return "\n".join(part.text if isinstance(part, TextContent) else f"[{type(part).__name__}]" for part in value)


def error_text(exc: BaseException) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class Fitter:
    """Makes frames fit the limit: a delta is split into several, a result is truncated with a marker.

    Text inside a frame is encoded twice, as a JSON string in the protocol's event and again in the frame's
    envelope, so a quote costs four bytes and a control character seven. Pieces are cut by an estimate of that cost
    and each one is then measured for real.
    """

    def __init__(self, max_frame_bytes: int, measure: Callable[[str], int]) -> None:
        self.limit = max_frame_bytes
        self.measure = measure
        self._costs: dict[str, int] = {}

    def fits(self, data: str) -> bool:
        return self.measure(data) <= self.limit

    def whole(self, data: str) -> str:
        """``data`` itself, which cannot be split: ``FrameTooLarge`` when it does not fit."""
        size = self.measure(data)
        if size > self.limit:
            raise FrameTooLarge(size, self.limit)
        return data

    def split(self, text: str, build: Callable[[str], str]) -> list[str]:
        """Pieces of ``text``, in order, each of whose frames ``build(piece)`` fits; they concatenate to ``text``."""
        if self.fits(build(text)):
            return [text]
        room = self.limit - self.measure(build(""))
        if room <= 0:
            raise FrameTooLarge(self.measure(build("")), self.limit)
        pieces: list[str] = []
        start = used = 0
        for index, char in enumerate(text):
            cost = self._cost(char)
            if used + cost > room and index > start:
                pieces.append(text[start:index])
                start, used = index, 0
            used += cost
        pieces.append(text[start:])
        return [part for piece in pieces for part in self._halves(piece, build)]

    def truncate(self, text: str, build: Callable[[str], str], max_bytes: int | None = None) -> str:
        """The longest start of ``text`` that keeps ``build(...)`` within the limit (and ``text`` within ``max_bytes``
        UTF-8 bytes), followed by a marker that gives the full length."""
        marker = f"\n[truncated, {len(text)} characters in total]"
        cut = text
        if max_bytes is not None and len(text.encode("utf-8", "surrogatepass")) > max_bytes:
            cut = text.encode("utf-8", "surrogatepass")[:max_bytes].decode("utf-8", "ignore")
        if cut == text and self.fits(build(text)):
            return text
        if not self.fits(build(marker)):
            raise FrameTooLarge(self.measure(build(marker)), self.limit)
        low, high = 0, len(cut)
        while low < high:
            middle = (low + high + 1) // 2
            if self.fits(build(cut[:middle] + marker)):
                low = middle
            else:
                high = middle - 1
        return cut[:low] + marker

    def _halves(self, piece: str, build: Callable[[str], str]) -> list[str]:
        if self.fits(build(piece)):
            return [piece]
        if len(piece) <= 1:
            raise FrameTooLarge(self.measure(build(piece)), self.limit)
        middle = len(piece) // 2
        return self._halves(piece[:middle], build) + self._halves(piece[middle:], build)

    def _cost(self, char: str) -> int:
        cost = self._costs.get(char)
        if cost is None:
            inner = _ENCODER.encode(char)[1:-1]
            cost = len(_ENCODER.encode(inner)[1:-1].encode("utf-8", "surrogatepass"))
            self._costs[char] = cost
        return cost


class ConverterOptions(TypedDict, total=False):
    """The options every built-in converter takes."""

    #: Publish tool results; ``False`` sends a placeholder in their place.
    include_tool_results: bool
    #: Tool results longer than this many UTF-8 bytes are truncated, with a marker that gives the full length.
    max_tool_result_bytes: int
    #: Publish error messages; by default a browser gets "An error occurred." in their place.
    expose_errors: bool
    #: The limit a frame must fit: the publisher's ``max_frame_bytes``.
    max_frame_bytes: int
    #: How a frame's data is measured: the publisher's ``frame_size``.
    measure: Callable[[str], int]


class BaseConverter:
    """The options every built-in converter takes, and their frame fitting."""

    def __init__(self, **options: Unpack[ConverterOptions]) -> None:
        max_tool_result_bytes = options.get("max_tool_result_bytes", 32_768)
        if max_tool_result_bytes < 1:
            raise ValueError("max_tool_result_bytes must be at least 1.")
        self.include_tool_results = options.get("include_tool_results", True)
        self.max_tool_result_bytes = max_tool_result_bytes
        self.expose_errors = options.get("expose_errors", False)
        self.fit = Fitter(options.get("max_frame_bytes", DEFAULT_MAX_FRAME_BYTES), options.get("measure", frame_size))

    def _error_message(self, exc: BaseException) -> str:
        return error_text(exc) if self.expose_errors else MASKED_ERROR

    def _result_content(self, result: ToolCallResult) -> str:
        """The text a browser gets for a tool result, before truncation."""
        if result.error and not self.expose_errors:
            return MASKED_ERROR
        if not self.include_tool_results:
            return OMITTED_RESULT
        return result_text(result)

    @staticmethod
    def _frames(data: Iterable[str]) -> list[Frame]:
        return [Frame(item) for item in data]
