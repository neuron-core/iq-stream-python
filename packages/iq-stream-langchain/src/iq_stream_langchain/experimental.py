"""A ``StreamTransformer`` for LangGraph's ``stream_events(version="v3")``, which LangGraph marks as beta.

It publishes the run's protocol events while the application reads the run's projections (``run.output``,
``run.messages``): the projections are single-consumer, so a wrapper around the stream would compete with them.
Needs ``langgraph`` 1.2 or later.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from iq_stream import AsyncPublisherLike, FrameTooLarge, PublisherLike

from ._encode import MASKED_MESSAGE, dumps, error_part
from ._relay import Oversize, logger, stopped

if TYPE_CHECKING:
    from langgraph.stream import ProtocolEvent, StreamTransformer  # pyright: ignore[reportMissingTypeStubs]

_STREAM_MODES = frozenset({"values", "updates", "checkpoints", "tasks", "debug", "messages", "custom", "tools"})


def _masked(data: Any) -> Any:
    if not isinstance(data, Mapping):
        return cast("object", data)
    fields = cast("Mapping[str, Any]", data)
    event = fields.get("event")
    if isinstance(event, str) and event.endswith("error") and "message" in fields:
        return {**fields, "message": MASKED_MESSAGE}
    if fields.get("error") is not None and "result" in fields:  # a task's result
        return {**fields, "error": MASKED_MESSAGE}
    return fields


def iq_transformer(
    pub: PublisherLike | AsyncPublisherLike,
    *,
    methods: Sequence[str] = ("messages", "tools", "updates", "custom"),
    expose_errors: bool = False,
    on_oversize: Callable[[Oversize], None] | None = None,
) -> Callable[[tuple[str, ...]], StreamTransformer]:
    """A transformer factory for ``graph.stream_events(input, version="v3", transformers=[...])``.

    Each protocol event of one of ``methods``, from every namespace, is one frame:
    ``{"type": "event", "method", "params": {"namespace", "timestamp", "data"}}``, with the metadata half of a
    ``messages`` payload left out. A run that fails publishes the same ``error`` part as ``relay()``. The
    transformer never closes the stream: the publisher's ``with`` block does, after ``run.output`` returned (or
    raised) and the application saved the answer. An event over the frame limit is not sent.
    """
    try:
        from langgraph.stream import StreamTransformer  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as error:  # langgraph < 1.2
        raise ImportError("iq_transformer() needs langgraph 1.2 or later, for stream_events(version='v3').") from error

    published = frozenset(methods)
    required = tuple(method for method in methods if method in _STREAM_MODES)
    reported: list[bool] = [False]

    class IQTransformer(StreamTransformer):
        required_stream_modes = required

        def __init__(self, scope: tuple[str, ...] = ()) -> None:
            super().__init__(scope)
            # The root mux sees the events of every namespace; a sub-graph's mux would repeat them.
            self.root = scope == ()

        def init(self) -> dict[str, Any]:
            return {}

        def process(self, event: ProtocolEvent) -> bool:
            if not self.root or event["method"] not in published or stopped(pub):
                return True
            params = event["params"]
            data: Any = params["data"]
            if event["method"] == "messages" and isinstance(data, (tuple, list)):
                data = cast("Sequence[Any]", data)[0]
            if not expose_errors:
                data = _masked(data)
            frame = dumps(
                {
                    "type": "event",
                    "method": event["method"],
                    "params": {"namespace": list(params["namespace"]), "timestamp": params["timestamp"], "data": data},
                }
            )
            try:
                pub.send(frame)
            except FrameTooLarge as error:
                report = Oversize(str(event["method"]), error.size, error.limit, truncated=False)
                if on_oversize is not None:
                    on_oversize(report)
                elif not reported[0]:
                    reported[0] = True
                    logger.warning(
                        "IQ stream %s: a %s event of %d bytes is over the frame limit of %d bytes and was not sent; "
                        "later ones are not reported.",
                        pub.stream_id,
                        report.part_type,
                        report.size,
                        report.limit,
                    )
            return True

        def fail(self, err: BaseException) -> None:
            if self.root and pub.alive:
                with contextlib.suppress(FrameTooLarge):
                    pub.send(error_part(err, expose=expose_errors))

    return IQTransformer
