"""Runs started in a task of their own, publishing as they go: what a route calls to answer with a ticket at once."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from uuid import uuid4

from iq_stream import IQClient, IQError, Outcome, Ticket

from ._common import AguiRequest, Protocol
from ._native import Transform
from ._tee import AgnoTrack, Options, Reporter, RunSummary, _Run, adrive, options  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.run.requirement import RunRequirement
    from agno.team import Team
    from agno.workflow import Workflow

logger = logging.getLogger("iq_stream_agno")

# The loop keeps only weak references to tasks: a run must not be collected while it publishes.
_running: set[asyncio.Task[RunSummary]] = set()


class PublishOptions(TypedDict, total=False):
    """The options of ``apublish()`` that ``start_run()``, ``continue_run()`` and the AgentOS interface pass on."""

    event_names: bool
    include: Collection[str] | None
    exclude: Collection[str]
    close_on_cancel: Outcome
    oversize: Literal["shrink", "drop"]
    transform: Transform | None
    expose_errors: bool
    on_dropped: Callable[[IQError], object] | None
    before_close: Callable[[RunSummary], Awaitable[object] | object] | None


@dataclass(frozen=True)
class StreamHandle:
    """A run publishing in a task of its own, and what its browser needs."""

    ticket: Ticket
    run_id: str
    session_id: str
    #: Its result is the run's ``RunSummary``. Cancelling it cancels the run and closes the stream with the
    #: ``close_on_cancel`` publish option (``fail`` by default).
    task: asyncio.Task[RunSummary] = field(repr=False)

    @property
    def stream_id(self) -> str:
        return self.ticket.stream_id

    @property
    def token(self) -> str:
        """The ``sub`` token."""
        return self.ticket.token

    @property
    def url(self) -> str:
        """The subscribe URL, token included."""
        return self.ticket.url

    @property
    def expires_at(self) -> int:
        return self.ticket.expires_at

    def to_dict(self) -> dict[str, str | int]:
        """What a route answers: the ticket's camelCase keys (``streamId``, ``token``, ``url``, ``expiresAt``), which
        the browser client of ``@neuron-core/iq-stream`` reads, with ``runId`` and ``sessionId``."""
        return {**self.ticket.to_dict(), "runId": self.run_id, "sessionId": self.session_id}


def running() -> frozenset[asyncio.Task[RunSummary]]:
    """The tasks of runs started here that have not finished yet."""
    return frozenset(_running)


def start_run(
    client: IQClient,
    entity: Agent | Team | Workflow,
    input: Any,
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    stream_id: str | None = None,
    ttl: float = 3600,
    protocol: Protocol = "agno",
    publish_options: PublishOptions | None = None,
    **run_kwargs: Any,
) -> StreamHandle:
    """Starts ``entity.arun(input, stream=True, stream_events=True, ...)`` in a task that publishes it, and returns at
    once. Called on a running event loop. The publisher is created first, so its request is open before Agno's
    setup begins; the run id is chosen here, so the run can be cancelled by it before its first event. The stream id is
    recorded in the run's metadata under ``iq_stream_id``."""
    return begin_run(
        client,
        entity,
        input,
        session_id=session_id,
        run_id=run_id,
        stream_id=stream_id,
        ttl=ttl,
        protocol=protocol,
        publish_options=publish_options,
        run_kwargs=run_kwargs,
    )


def begin_run(
    client: IQClient,
    entity: Agent | Team | Workflow,
    input: Any,
    *,
    session_id: str | None,
    run_id: str | None,
    stream_id: str | None,
    ttl: float,
    protocol: Protocol,
    publish_options: PublishOptions | None,
    run_kwargs: dict[str, Any],
    on_settled: Callable[[StreamHandle], object] | None = None,
) -> StreamHandle:
    """``start_run()``, with what the AgentOS interface needs: ``on_settled``, called when the run's ending has
    passed (``_Run.on_settled``)."""
    run_id = run_id or str(uuid4())
    session_id = session_id or str(uuid4())
    checked = checked_options(protocol, publish_options, run_id)
    ticket = client.ticket(stream_id, ttl)
    metadata: dict[str, Any] = {**(run_kwargs.pop("metadata", None) or {}), "iq_stream_id": ticket.stream_id}

    target: Any = entity  # the three classes' overloads differ; each takes these arguments

    def events() -> AsyncIterable[object]:
        return target.arun(
            input,
            stream=True,
            stream_events=True,
            run_id=run_id,
            session_id=session_id,
            metadata=metadata,
            **run_kwargs,
        )

    before_close = (publish_options or {}).get("before_close")
    return launch(client, ticket, events, run_id, session_id, checked, before_close, on_settled=on_settled)


def continue_run(
    client: IQClient,
    entity: Agent | Team,
    *,
    run_id: str,
    session_id: str,
    requirements: list[RunRequirement] | None = None,
    stream_id: str | None = None,
    ttl: float = 3600,
    protocol: Protocol = "agno",
    publish_options: PublishOptions | None = None,
    **run_kwargs: Any,
) -> StreamHandle:
    """One more leg of a paused run, published on a new stream: ``entity.acontinue_run(run_id=..., requirements=...,
    stream=True, stream_events=True)``. The first leg's stream closed at the pause; the run keeps its id."""
    return begin_continue(
        client,
        entity,
        run_id=run_id,
        session_id=session_id,
        requirements=requirements,
        stream_id=stream_id,
        ttl=ttl,
        protocol=protocol,
        publish_options=publish_options,
        run_kwargs=run_kwargs,
    )


def begin_continue(
    client: IQClient,
    entity: Agent | Team,
    *,
    run_id: str,
    session_id: str,
    requirements: list[RunRequirement] | None,
    stream_id: str | None,
    ttl: float,
    protocol: Protocol,
    publish_options: PublishOptions | None,
    run_kwargs: dict[str, Any],
    on_settled: Callable[[StreamHandle], object] | None = None,
) -> StreamHandle:
    checked = checked_options(protocol, publish_options, run_id)
    ticket = client.ticket(stream_id, ttl)

    target: Any = entity

    def events() -> AsyncIterable[object]:
        return target.acontinue_run(
            run_id=run_id,
            session_id=session_id,
            requirements=requirements,
            stream=True,
            stream_events=True,
            **run_kwargs,
        )

    before_close = (publish_options or {}).get("before_close")
    return launch(client, ticket, events, run_id, session_id, checked, before_close, on_settled=on_settled)


def checked_options(protocol: Protocol, publish_options: PublishOptions | None, root_run_id: str | None) -> Options:
    chosen: PublishOptions = publish_options or {}
    return options(
        protocol=protocol,
        event_names=chosen.get("event_names", False),
        include=chosen.get("include"),
        exclude=chosen.get("exclude", ()),
        root_run_id=root_run_id,
        close_on_cancel=chosen.get("close_on_cancel", "fail"),
        oversize=chosen.get("oversize", "shrink"),
        transform=chosen.get("transform"),
        expose_errors=chosen.get("expose_errors", False),
        on_dropped=chosen.get("on_dropped"),
        hold_ending=chosen.get("before_close") is not None,
    )


def launch(
    client: IQClient,
    ticket: Ticket,
    events: Callable[[], AsyncIterable[object]],
    run_id: str,
    session_id: str,
    checked: Options,
    before_close: Callable[[RunSummary], Awaitable[object] | object] | None,
    agui: AguiRequest | None = None,
    on_settled: Callable[[StreamHandle], object] | None = None,
) -> StreamHandle:
    publisher = client.apublisher(ticket.stream_id)  # its request opens before Agno's setup starts
    run = _Run(publisher, AgnoTrack(checked, publisher, Reporter(checked.on_dropped), agui), checked.close_on_cancel)
    run.track.session_id = session_id
    task = asyncio.get_running_loop().create_task(
        publish_task(run, events, before_close), name=f"iq-stream-agno {run_id}"
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    handle = StreamHandle(ticket, run_id, session_id, task)
    if on_settled is not None:
        run.on_settled = lambda: on_settled(handle)  # the task has not started: it runs at the loop's next turn
    return handle


async def publish_task(
    run: _Run,
    events: Callable[[], AsyncIterable[object]],
    before_close: Callable[[RunSummary], Awaitable[object] | object] | None,
) -> RunSummary:
    """Drives one run to its end. An error is the summary's, not the task's, so that a task nobody awaits leaves
    nothing unretrieved; a cancellation is the task's."""
    summaries: list[RunSummary] = []
    try:
        async for _ in adrive(events, run, before_close, summaries.append):
            pass
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error("The run ended with an exception", exc_info=error)
    return summaries[-1]
