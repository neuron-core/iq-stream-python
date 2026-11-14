"""Publish LlamaIndex workflow and agent runs to browsers through an IQ-Stream server.

``publish(handler, publisher)`` is the one reader of ``handler.stream_events()``: it turns each event into frames,
awaits the run, lets the application save the answer, and closes the stream with ``end`` or ``fail``.
``published()`` does the same and hands each event on. ``NativeEncoder`` and ``AGUIEncoder`` are the wire formats.
"""

from ._outcome import RunOutcome, RunStatus
from ._publish import PublishedStream, publish, published
from .encoders import DEFAULT_MAX_FIELD_BYTES, MASKED_ERROR, AGUIEncoder, Encoder, Fits, Frame, NativeEncoder

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MAX_FIELD_BYTES",
    "MASKED_ERROR",
    "AGUIEncoder",
    "Encoder",
    "Fits",
    "Frame",
    "NativeEncoder",
    "PublishedStream",
    "RunOutcome",
    "RunStatus",
    "__version__",
    "publish",
    "published",
]
