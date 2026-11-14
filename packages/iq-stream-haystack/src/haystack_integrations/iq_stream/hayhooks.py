"""Start a run in the background and answer with its ticket at once: for a Hayhooks pipeline wrapper's
``run_api_async`` (``start_stream``) or ``run_api`` (``start_stream_sync``), or any other server route.

The run is published from the serving process, in a task or a thread of its own, inside an ``IQStream`` block;
``on_result`` is where the application saves the result, before the stream's close line. These helpers do not
import Hayhooks, so they serve FastAPI routes just as well.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import StreamingCallbackT
from iq_stream import IQClient

from .stream import IQStream

logger = logging.getLogger("haystack_integrations.iq_stream")

Target = Agent | Pipeline
OnResult = Callable[[str, dict[str, Any]], Awaitable[None] | None]

# The event loop keeps only weak references to tasks: these keep the runs alive until they end.
_TASKS: set[asyncio.Task[None]] = set()


async def start_stream(
    client: IQClient,
    target: Target,
    run_args: Mapping[str, Any],
    *,
    on_result: OnResult | None = None,
    ttl: float = 3600,
    **stream_options: Any,
) -> dict[str, str | int]:
    """Starts ``target`` in a task of the running loop and returns the ticket's ``to_dict()`` (``streamId``,
    ``token``, ``url``, ``expiresAt``) without waiting for it.

    ``run_args`` are the keyword arguments of ``Agent.run_async``, or the data of ``Pipeline.run_async`` in its
    nested form (``{"component": {"input": ...}}``); the stream's callback is given to the agent, or to every
    component of the pipeline that takes a ``streaming_callback``. ``on_result(stream_id, result)``, sync or async,
    runs inside the stream's block, before its close line. ``stream_options`` go to ``IQStream`` (``stream_id``,
    ``protocol``, ...). An exception in the run closes the stream with ``fail`` and is logged.
    """
    stream = IQStream(client, **stream_options)
    ticket = client.ticket(stream.stream_id, ttl)
    task = asyncio.create_task(_run_async(stream, target, dict(run_args), on_result))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return ticket.to_dict()


def start_stream_sync(
    client: IQClient,
    target: Target,
    run_args: Mapping[str, Any],
    *,
    on_result: Callable[[str, dict[str, Any]], object] | None = None,
    ttl: float = 3600,
    **stream_options: Any,
) -> dict[str, str | int]:
    """``start_stream`` for synchronous code: ``target.run()`` in a thread of its own, under a copy of the caller's
    context, and ``on_result`` called in that thread."""
    stream = IQStream(client, **stream_options)
    ticket = client.ticket(stream.stream_id, ttl)
    context = contextvars.copy_context()
    thread = threading.Thread(
        target=context.run,
        args=(_run_sync, stream, target, dict(run_args), on_result),
        name=f"iq-stream-haystack {stream.stream_id}",
        daemon=True,
    )
    thread.start()
    return ticket.to_dict()


async def _run_async(stream: IQStream, target: Target, run_args: dict[str, Any], on_result: OnResult | None) -> None:
    try:
        async with stream:
            if isinstance(target, Agent):
                callback = _callback(stream, target, run_args.pop("streaming_callback", None))
                result = await target.run_async(**run_args, streaming_callback=callback)
            else:
                result = await target.run_async(_with_callbacks(stream, target, run_args))
            if on_result is not None:
                saved = on_result(stream.stream_id, result)
                if inspect.isawaitable(saved):
                    await saved
    except Exception:
        logger.exception("IQ stream %s: the run failed.", stream.stream_id)


def _run_sync(
    stream: IQStream,
    target: Target,
    run_args: dict[str, Any],
    on_result: Callable[[str, dict[str, Any]], object] | None,
) -> None:
    try:
        with stream:
            if isinstance(target, Agent):
                callback = _callback(stream, target, run_args.pop("streaming_callback", None), sync=True)
                result = target.run(**run_args, streaming_callback=callback)
            else:
                result = target.run(_with_callbacks(stream, target, run_args, sync=True))
            if on_result is not None:
                on_result(stream.stream_id, result)
    except Exception:
        logger.exception("IQ stream %s: the run failed.", stream.stream_id)


def _callback(stream: IQStream, component: object, theirs: Any = None, *, sync: bool = False) -> StreamingCallbackT:
    """The stream's callback for a component: async where Haystack awaits it, which an agent does only when its chat
    generator has ``run_async``; teed with the component's own callback if the data gave one."""
    inner = component.chat_generator if isinstance(component, Agent) else component
    if not sync and getattr(inner, "__haystack_supports_async__", False):
        return stream.acallback if theirs is None else stream.tee(cast(StreamingCallbackT, theirs))
    return stream.callback if theirs is None else stream.tee(cast(StreamingCallbackT, theirs))


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
