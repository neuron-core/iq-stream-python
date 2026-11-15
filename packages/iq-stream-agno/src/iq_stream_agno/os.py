"""An AgentOS interface whose routes start a run and answer with a ticket at once; the run's events reach the browser
through IQ-Stream, from a task that publishes them. Needs the ``os`` extra (and ``ag-ui`` for the AG-UI route)."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

try:
    from agno.os.interfaces.base import BaseInterface
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
except ImportError as error:  # pragma: no cover - depends on the installed extras
    raise ImportError(
        f"iq_stream_agno.os needs the os extra: pip install 'iq-stream-agno[os]' ({error.name or error} is missing)."
    ) from error

from iq_stream import IQClient

from ._common import AguiRequest, Protocol
from ._runner import PublishOptions, StreamHandle, begin_continue, begin_run, checked_options, launch
from ._tee import RunSummary

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.team import Team
    from agno.workflow import Workflow

__all__ = ["ContinueRequest", "IQStream", "RunRequest"]

logger = logging.getLogger("iq_stream_agno")


class RunRequest(BaseModel):
    """The body of ``POST {prefix}/runs``."""

    message: str
    session_id: str | None = None
    #: Taken only from an anonymous caller: an authenticated one runs as itself, as on Agno's own routes.
    user_id: str | None = None


class ContinueRequest(BaseModel):
    """The body of ``POST {prefix}/runs/{run_id}/continue``: the paused tools with the human's answers, as the
    ``RunPaused`` event listed them (``confirmed``, ``user_input_schema`` values, an external ``result``)."""

    session_id: str
    tools: list[dict[str, Any]] | None = None
    user_id: str | None = None


class IQStream(BaseInterface):
    """Mounts, under ``prefix``:

    - ``POST /runs`` — ``{"message", "session_id"?, "user_id"?}``; starts the run and answers with its ticket.
    - ``POST /agui`` — an AG-UI ``RunAgentInput``, for an agent or a team when ``ag-ui-protocol`` is installed: read
      as Agno's own AG-UI route reads it, and published in AG-UI as ``apublish(protocol="ag-ui")`` publishes.
    - ``POST /runs/{run_id}/continue`` — the next leg of a paused run, on a new stream.
    - ``POST /runs/{run_id}/cancel`` — cancels a run this interface is publishing.

    A ticket is ``{"streamId", "token", "url", "expiresAt", "runId", "sessionId"}``. A run id is refused (409) while
    its run has not reached its ending here; a paused run may be continued as soon as its pause is published.
    """

    type = "iq-stream"

    def __init__(
        self,
        client: IQClient,
        *,
        agent: Agent | None = None,
        team: Team | None = None,
        workflow: Workflow | None = None,
        prefix: str = "/iq",
        tags: list[str] | None = None,
        protocol: Protocol = "agno",
        ttl: float = 3600,
        publish_options: PublishOptions | None = None,
        shutdown_timeout: float = 10.0,
    ) -> None:
        if sum(entity is not None for entity in (agent, team, workflow)) != 1:
            raise ValueError("IQStream publishes one agent, team or workflow.")
        # Checked here, so that a bad option fails at startup rather than in a route or at a run's close line.
        self._checked = checked_options(protocol, publish_options, None)
        self.client = client
        self.entity: Agent | Team | Workflow = agent or team or workflow  # type: ignore[assignment]
        self.agent = agent  # pyright: ignore[reportIncompatibleVariableOverride]
        self.team = team  # pyright: ignore[reportIncompatibleVariableOverride]
        self.workflow = workflow  # pyright: ignore[reportIncompatibleVariableOverride]
        self.prefix = prefix
        self.tags = tags or ["IQ-Stream"]
        self.protocol: Protocol = protocol
        self.ttl = ttl
        self.publish_options: PublishOptions = publish_options or {}
        self.shutdown_timeout = shutdown_timeout
        # Runs this interface is publishing whose ending has not passed yet: who started each, so that only they can
        # cancel it. A run leaves when its ending passes, not when its close line is answered (held back until the
        # stream is min_open old), so that a browser may continue a paused run as soon as it reads the pause.
        self._runs: dict[str, tuple[StreamHandle, str | None]] = {}
        # Every task still publishing, closing ones included, which a shutdown waits for.
        self._tasks: set[asyncio.Task[RunSummary]] = set()

    @property
    def _family(self) -> Literal["agents", "teams", "workflows"]:
        return "agents" if self.agent is not None else "teams" if self.team is not None else "workflows"

    def get_router(self, use_async: bool = True, **kwargs: Any) -> APIRouter:
        router = APIRouter(prefix=self.prefix, tags=list(self.tags))  # pyright: ignore[reportArgumentType]

        @router.post("/runs")
        async def start(request: Request, body: RunRequest) -> dict[str, str | int]:  # pyright: ignore[reportUnusedFunction]
            user_id = _resolve_user(request, body.user_id)
            session_id = body.session_id or str(uuid4())
            await _assert_session_writable(self.entity, session_id, user_id, request)
            handle = begin_run(
                self.client,
                self.entity,
                body.message,
                session_id=session_id,
                run_id=None,
                stream_id=None,
                ttl=self.ttl,
                protocol=self.protocol,
                publish_options=self.publish_options,
                run_kwargs={"user_id": user_id},
                on_settled=self._settled,
            )
            self._keep(handle, user_id)
            return handle.to_dict()

        @router.post("/runs/{run_id}/continue")
        async def resume(request: Request, run_id: str, body: ContinueRequest) -> dict[str, str | int]:  # pyright: ignore[reportUnusedFunction]
            entity: Any = self.agent or self.team
            if entity is None:
                raise HTTPException(status_code=400, detail="Only an agent's or a team's run is continued here.")
            user_id = _resolve_user(request, body.user_id)
            stored: object = await entity.aget_run_output(run_id=run_id, session_id=body.session_id)
            if stored is None or getattr(stored, "user_id", None) != user_id:
                raise HTTPException(status_code=404, detail="Run not found")
            reason = await _continuation_blocked(entity, run_id, request)
            if reason:
                raise HTTPException(status_code=403, detail=reason)
            self._refuse_running(run_id)  # with no await between it and _keep()
            handle = begin_continue(
                self.client,
                entity,
                run_id=run_id,
                session_id=body.session_id,
                requirements=_requirements(body.tools),
                stream_id=None,
                ttl=self.ttl,
                protocol=self.protocol,
                publish_options=self.publish_options,
                run_kwargs={"user_id": user_id},
                on_settled=self._settled,
            )
            self._keep(handle, user_id)
            return handle.to_dict()

        @router.post("/runs/{run_id}/cancel")
        async def cancel(request: Request, run_id: str, user_id: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
            kept = self._runs.get(run_id)
            # Only who started the run, resolved as at the start (user_id in the query is for anonymous callers).
            if kept is None or kept[1] != _resolve_user(request, user_id):
                raise HTTPException(status_code=404, detail="Run not found")
            await _cancel(self.entity, run_id)
            return {"runId": run_id}

        if self.workflow is None and _has_agui():
            self._mount_agui(router)
        self.router = router
        return router

    def _mount_agui(self, router: APIRouter) -> None:
        from ag_ui.core import RunAgentInput

        async def agui(request: Request, run_input: RunAgentInput) -> dict[str, str | int]:
            props: object = run_input.forwarded_props
            client_user_id = cast("dict[str, object]", props).get("user_id") if isinstance(props, dict) else None
            user_id = _resolve_user(request, client_user_id if isinstance(client_user_id, str) else None)
            await _assert_session_writable(self.entity, run_input.thread_id, user_id, request)
            handle = await self._start_agui(run_input, user_id, request)
            self._keep(handle, user_id)
            return handle.to_dict()

        # FastAPI reads annotations in the module's globals, where ag_ui is not imported: give it the class itself.
        agui.__annotations__["run_input"] = RunAgentInput
        router.post("/agui")(agui)

    async def _start_agui(self, run_input: Any, user_id: str | None, request: Request) -> StreamHandle:
        """What Agno's AG-UI route does (``run_entity()``) up to the conversion, with its own helpers: the inputs read
        from the request, then a new run, or the paused run that trailing tool results answer, which Agno continues
        under that run's id. The events then go through the tee, as with ``apublish(protocol="ag-ui")``: masked, sized
        and closed like those of every other route."""
        from agno.os.interfaces.agui.input import (
            extract_context,
            extract_media,
            extract_tool_messages,
            extract_user_input,
            parse_client_tools,
            validate_state,
        )
        from agno.os.interfaces.agui.resume import resume_paused_run  # pyright: ignore[reportUnknownVariableType]
        from agno.run.base import RunContext

        entity: Any = self.entity
        thread_id: str = run_input.thread_id
        # As run_entity(): RunAgentInput takes an empty runId, and Agno would give the run another id than the root's.
        request_run_id: str = run_input.run_id or str(uuid4())
        messages: list[Any] = run_input.messages or []
        state = validate_state(run_input.state, thread_id)
        dependencies = extract_context(run_input.context)
        context = RunContext(
            run_id=request_run_id,
            session_id=thread_id,
            user_id=user_id,
            client_tools=parse_client_tools(run_input.tools) or None,
            dependencies=dependencies,
            session_state=state,
        )
        extra: dict[str, Any] = {"add_dependencies_to_context": True} if dependencies else {}
        tool_messages = extract_tool_messages(messages)
        if tool_messages:
            try:
                resumed: AsyncIterable[object] = await resume_paused_run(
                    entity=entity,
                    session_id=thread_id,
                    tool_messages=tool_messages,
                    run_context=context,
                    run_kwargs=extra,
                )
            except ValueError as error:  # no paused run answers these tool results, or it cannot be resumed here
                raise HTTPException(status_code=400, detail=str(error)) from error
            run_id: str = context.run_id
            # Refused while an admin approval is pending, as Agno's own AG-UI route refuses it.
            reason = await _continuation_blocked(entity, run_id, request)
            if reason:
                raise HTTPException(status_code=403, detail=reason)

            def events() -> AsyncIterable[object]:
                return resumed

        else:
            run_id = request_run_id
            images, audio, videos, files = extract_media(messages)
            if _is_remote(entity):
                # As run_entity(): a RunContext does not cross the wire, so the state and the context go as fields.
                extra.update(session_state=state, dependencies=dependencies)
                if context.client_tools:
                    logger.warning("AG-UI client tools are not forwarded to remote agents or teams")
            else:
                extra["run_context"] = context

            def events() -> AsyncIterable[object]:
                return entity.arun(
                    input=extract_user_input(messages),
                    stream=True,
                    stream_events=True,
                    session_id=thread_id,
                    user_id=user_id,
                    run_id=run_id,
                    images=images or None,
                    audio=audio or None,
                    videos=videos or None,
                    files=files or None,
                    **extra,
                )

        self._refuse_running(run_id)  # with no await between it and _keep()
        checked = dataclasses.replace(
            self._checked, protocol="ag-ui", event_names=False, transform=None, root_run_id=run_id
        )
        ticket = self.client.ticket(None, self.ttl)
        before_close = self.publish_options.get("before_close")
        return launch(
            self.client,
            ticket,
            events,
            run_id,
            thread_id,
            checked,
            before_close,
            AguiRequest(thread_id, request_run_id, state),
            on_settled=self._settled,
        )

    def _refuse_running(self, run_id: str) -> None:
        """A run id names one run in flight here, until its ending passes: a second request for it would take its
        place in the registry."""
        if run_id in self._runs:
            raise HTTPException(status_code=409, detail="The run is already in flight")

    def _keep(self, handle: StreamHandle, user_id: str | None) -> None:
        self._runs[handle.run_id] = (handle, user_id)
        self._tasks.add(handle.task)

        def forget(task: asyncio.Task[RunSummary]) -> None:
            self._tasks.discard(task)
            self._settled(handle)

        handle.task.add_done_callback(forget)

    def _settled(self, handle: StreamHandle) -> None:
        """The run's ending passed: it is no longer in flight here, though its stream may still be closing."""
        if self._runs.get(handle.run_id, (None,))[0] is handle:
            del self._runs[handle.run_id]

    def get_scope_mappings(self) -> dict[str, list[str]]:
        scope = [f"{self._family}:run"]
        routes = ["runs", "runs/*/continue", "runs/*/cancel"]
        if self.workflow is None and _has_agui():
            routes.append("agui")
        return {f"POST {self.prefix}/{route}": scope for route in routes}

    async def astart(self) -> None:
        return None

    async def aclose(self) -> None:
        """AgentOS's shutdown (agno 3): waits up to ``shutdown_timeout`` for the runs it publishes, streams still
        closing included, then cancels the rest, whose streams close with ``close_on_cancel`` (``fail`` by default).
        On agno 2.x, which has no such hook, call it from your lifespan."""
        tasks = list(self._tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self.shutdown_timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=15.0)


def _has_agui() -> bool:
    try:
        import ag_ui.core  # noqa: F401  # pyright: ignore[reportUnusedImport]
        import agno.os.interfaces.agui.router  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError:
        return False
    return True


def _is_remote(entity: object) -> bool:
    """Whether the entity runs in another AgentOS: ``RemoteAgent`` or ``RemoteTeam``."""
    try:
        from agno.agent import RemoteAgent
        from agno.team.remote import RemoteTeam
    except ImportError:  # pragma: no cover - both are in agno 2.9 and later
        return False
    return isinstance(entity, (RemoteAgent, RemoteTeam))


def _resolve_user(request: Request, client_user_id: str | None) -> str | None:
    """The identity a run is attributed to, as Agno's own run routes resolve it: an authenticated caller runs as
    itself, an anonymous one may name a user."""
    from agno.os.middleware.user_scope import resolve_run_user_id

    return resolve_run_user_id(request, client_user_id)


async def _assert_session_writable(entity: Any, session_id: str, user_id: str | None, request: Request) -> None:
    """Refuses a session that belongs to another user, as Agno's routes do where the installed Agno can tell."""
    try:
        from agno.os.middleware.user_scope import (  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
            assert_session_writable,
            caller_is_admin,
        )
    except ImportError:  # agno 2.x checks nothing here either
        return
    await assert_session_writable(
        getattr(entity, "db", None),
        session_id,
        user_id or getattr(entity, "user_id", None),
        is_admin=caller_is_admin(request),
    )


async def _continuation_blocked(entity: Any, run_id: str, request: Request) -> str | None:
    from agno.os.auth import run_continuation_blocked_reason

    extra: dict[str, Any] = {}
    if "request" in inspect.signature(run_continuation_blocked_reason).parameters:  # agno 3
        extra["request"] = request
    reason: object = await run_continuation_blocked_reason(
        getattr(entity, "db", None),
        run_id,
        authorization_enabled=bool(getattr(request.state, "authorization_enabled", False)),
        user_scopes=list(getattr(request.state, "scopes", None) or []),
        **extra,
    )
    return reason if isinstance(reason, str) and reason else None


def _requirements(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    """The paused tools as AgentOS's own continue route turns them into requirements (``updated_tools``, which it
    passes too, is deprecated since agno 2.9)."""
    if not tools:
        return None
    from agno.models.response import ToolExecution
    from agno.run.requirement import RunRequirement

    try:
        return [RunRequirement(tool_execution=ToolExecution.from_dict(tool)) for tool in tools]
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid structure or content for tools") from error


async def _cancel(entity: Any, run_id: str) -> None:
    """Sets Agno's cancellation flag; the run then ends with ``RunCancelled`` and its stream with ``fail``."""
    await entity.acancel_run(run_id)
