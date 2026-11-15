"""The wire formats: ``NativeEncoder`` (the event class as SSE event name, JSON data) and ``AGUIEncoder`` (AG-UI
events). An encoder of one's own implements the ``Encoder`` protocol."""

from .agui import AGUIEncoder
from .base import DEFAULT_MAX_FIELD_BYTES, MASKED_ERROR, Encoder, Fits, Frame
from .native import NativeEncoder

__all__ = ["DEFAULT_MAX_FIELD_BYTES", "MASKED_ERROR", "AGUIEncoder", "Encoder", "Fits", "Frame", "NativeEncoder"]
