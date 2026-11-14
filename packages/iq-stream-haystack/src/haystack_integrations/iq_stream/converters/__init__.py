"""Converters from Haystack's ``StreamingChunk`` to the frames a browser reads: AG-UI events, AI SDK UI message
chunks, or the chunks' own JSON. ``IQStream`` builds the one its ``protocol`` names; any object that follows
``ChunkConverter`` can take its place."""

from .agui import AGUIConverter
from .ai_sdk import AISDKConverter
from .base import ChunkConverter, ConverterOptions, Frame
from .native import NativeConverter

__all__ = ["AGUIConverter", "AISDKConverter", "ChunkConverter", "ConverterOptions", "Frame", "NativeConverter"]
