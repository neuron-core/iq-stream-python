"""Module-level streaming callbacks. A component built with one of them serializes (``to_dict()``, YAML) and loads
without configuration, since Haystack allows the ``haystack_integrations`` namespace by default; at run time the
callback publishes to the ``IQStream`` whose block is active in the calling context, and does nothing outside one."""

from __future__ import annotations

from haystack.dataclasses import StreamingChunk

from .stream import current_stream


def iq_streaming_callback(chunk: StreamingChunk) -> None:
    """Publishes ``chunk`` to the active ``IQStream``, if there is one. For ``run()``."""
    stream = current_stream()
    if stream is not None:
        stream.callback(chunk)


async def iq_streaming_callback_async(chunk: StreamingChunk) -> None:
    """Publishes ``chunk`` to the active ``IQStream``, if there is one. For ``run_async()``."""
    iq_streaming_callback(chunk)
