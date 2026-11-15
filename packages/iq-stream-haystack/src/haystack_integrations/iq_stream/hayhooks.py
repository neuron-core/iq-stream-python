"""Start a run in the background and answer with its ticket at once: for a Hayhooks pipeline wrapper's
``run_api_async`` (``start_stream``) or ``run_api`` (``start_stream_sync``), or any other server route.

The run is published from the serving process, in a task or a thread of its own, inside an ``IQStream`` block;
``before_close`` is where the application saves the result, before the stream's close line. These helpers do not
import Hayhooks, so they serve FastAPI routes just as well.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypedDict, cast

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import StreamingCallbackT
from iq_stream import IQClient, IQError, Outcome

from .converters import ChunkConverter
from .stream import IQStream, ProtocolName

if TYPE_CHECKING:
    from typing_extensions import Unpack

logger = logging.getLogger("haystack_integrations.iq_stream")

Target = Agent | Pipeline
BeforeClose = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class StreamOptions(TypedDict, total=False):
    """The options of ``IQStream`` that the helpers pass on."""

    stream_id: str | None
    protocol: ProtocolName | ChunkConverter
    thread_id: str | None
    run_id: str | None
    include_tool_results: bool
    max_tool_result_bytes: int
    expose_errors: bool
    close_on_cancel: Outcome
    close_timeout: float
    on_error: Callable[[IQError], object] | None


# The event loop keeps only weak references to tasks: these keep the runs alive until they end.
_TASKS: set[asyncio.Task[None]] = set()


async def start_stream(
    client: IQClient,
    target: Target,
    run_args: Mapping[str, Any],
    *,
    before_close: BeforeClose | None = None,
    ttl: float = 3600,
    **stream_options: Unpack[StreamOptions],
) -> dict[str, str | int]:
    """Starts ``target`` in a task of the running loop and returns the ticket's ``to_dict()`` (``streamId``,
    ``token``, ``url``, ``expiresAt``) without waiting for it.

    ``run_args`` are the keyword arguments of ``Agent.run_async``, or the data of ``Pipeline.run_async`` in its
    nested form (``{"component": {"input": ...}}``); the stream's callback is given to the agent, or to every
    component of the pipeline that takes a ``streaming_callback``. ``before_close(stream_id, result)``, sync or
    async, runs inside the stream's block, before its close line. ``stream_options`` go to ``IQStream``
    (``stream_id``, ``protocol``, ``thread_id``, ``close_on_cancel``, ...). An exception in the run closes the stream
    with ``fail`` and is logged; so is a publishing failure that leaves the block with a client opened with
    ``on_error="raise"``.
    """
    stream = IQStream(client, **stream_options)
    kwargs = _arguments(stream, target, run_args, sync=False)
    ticket = client.ticket(stream.stream_id, ttl)
    task = asyncio.create_task(_run_async(stream, target, kwargs, before_close))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return ticket.to_dict()


def start_stream_sync(
    client: IQClient,
    target: Target,
    run_args: Mapping[str, Any],
    *,
    before_close: Callable[[str, dict[str, Any]], object] | None = None,
    ttl: float = 3600,
    **stream_options: Unpack[StreamOptions],
) -> dict[str, str | int]:
    """``start_stream`` for synchronous code: ``target.run()`` in a thread of its own, under a copy of the caller's
    context, and ``before_close`` called in that thread."""
    stream = IQStream(client, **stream_options)
    kwargs = _arguments(stream, target, run_args, sync=True)
    ticket = client.ticket(stream.stream_id, ttl)
    context = contextvars.copy_context()
    thread = threading.Thread(
        target=context.run,
        args=(_run_sync, stream, target, kwargs, before_close),
        name=f"iq-stream-haystack {stream.stream_id}",
        daemon=True,
    )
    thread.start()
    return ticket.to_dict()


def _arguments(stream: IQStream, target: Target, run_args: Mapping[str, Any], *, sync: bool) -> dict[str, Any]:
    """The keyword arguments of ``target.run`` or ``run_async``, with the stream's callback in place."""
    if isinstance(target, Agent):
        kwargs = dict(run_args)
        kwargs["streaming_callback"] = _callback(stream, target, kwargs.get("streaming_callback"), sync=sync)
        return kwargs
    if isinstance(target, Pipeline):  # pyright: ignore[reportUnnecessaryIsInstance]
        return {"data": _with_callbacks(stream, target, dict(run_args), sync=sync)}
    raise TypeError("target must be a Haystack Agent or Pipeline.")


async def _run_async(
    stream: IQStream, target: Target, kwargs: dict[str, Any], before_close: BeforeClose | None
) -> None:
    ran = False
    try:
        async with stream:
            result = await target.run_async(**kwargs)
            if before_close is not None:
                saved = before_close(stream.stream_id, result)
                if inspect.isawaitable(saved):
                    await saved
            ran = True
    except Exception as error:
        _log_failure(stream, error, ran)


def _run_sync(
    stream: IQStream,
    target: Target,
    kwargs: dict[str, Any],
    before_close: Callable[[str, dict[str, Any]], object] | None,
) -> None:
    ran = False
    try:
        with stream:
            result = target.run(**kwargs)
            if before_close is not None:
                before_close(stream.stream_id, result)
            ran = True
    except Exception as error:
        _log_failure(stream, error, ran)


def _log_failure(stream: IQStream, error: Exception, ran: bool) -> None:
    if ran and error is stream.error:  # leaving raised the publisher's IQError: its client has on_error="raise"
        logger.warning("%s; the run itself completed, and before_close ran.", error)
    else:
        logger.exception("IQ stream %s: the run failed.", stream.stream_id)


def _callback(stream: IQStream, component: object, theirs: Any = None, *, sync: bool = False) -> StreamingCallbackT:
    """The stream's callback for a component, teed with the component's own callback if the data gave one: async
    only where every generator that gets it awaits it, since Haystack refuses a coroutine callback where it runs a
    generator in a thread; the sync one works everywhere, at the cost of Haystack's warning."""
    if theirs is not None:
        return stream.tee(cast(StreamingCallbackT, theirs))
    return stream.acallback if not sync and _awaits_callback(component) else stream.callback


def _awaits_callback(component: object) -> bool:
    """Whether every chat generator that would get the callback from ``component`` in ``run_async`` has ``run_async``
    of its own. False when that is not known: an agent that passes its callback to tools hands it to sub-agents
    and functions whose way of calling it cannot be seen, whether they were given at construction or at run time."""
    if isinstance(component, Agent):
        if component.tool_streaming_callback_passthrough:
            return False
        return _awaits_callback(component.chat_generator)
    if not getattr(component, "__haystack_supports_async__", False):
        return False
    wrapped: object = getattr(component, "chat_generators", ())  # FallbackChatGenerator's, run each in its own way
    return all(_awaits_callback(generator) for generator in cast(list[object], wrapped))


def _with_callbacks(
    stream: IQStream, pipeline: Pipeline, data: dict[str, Any], *, sync: bool = False
) -> dict[str, Any]:
    sockets = pipeline.inputs(include_components_with_connected_inputs=True)
    if not set(data) <= set(sockets):
        raise ValueError("Pass the pipeline's data in its nested form: {component_name: {input_name: value}}.")
    data = {name: dict(inputs) for name, inputs in data.items()}
    for name, inputs_of in sockets.items():
        if "streaming_callback" in inputs_of:
            inputs = data.setdefault(name, {})
            instance = pipeline.get_component(name)
            inputs["streaming_callback"] = _callback(stream, instance, inputs.get("streaming_callback"), sync=sync)
    return data
