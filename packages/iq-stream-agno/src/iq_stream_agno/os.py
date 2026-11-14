"""An AgentOS interface whose routes start a run and answer with a ticket at once; the run's events reach the browser
through IQ-Stream, from a task that publishes them. Needs the ``os`` extra (and ``agui`` for the AG-UI route)."""

from __future__ import annotations

import asyncio
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

from ._common import Protocol
from ._runner import PublishOptions, StreamHandle, _running, continue_run, publish_task, start_run  # pyright: ignore[reportPrivateUsage]
from ._tee import RunSummary, _Run  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.team import Team
    from agno.workflow import Workflow

__all__ = ["ContinueRequest", "IQStream", "RunRequest"]


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
    - ``POST /agui`` — an AG-UI ``RunAgentInput``, for an agent or a team when ``ag-ui-protocol`` is installed;
      Agno's own AG-UI conversion (``run_entity``) is published, and the answer is the ticket.
    - ``POST /runs/{run_id}/continue`` — the next leg of a paused run, on a new stream.
    - ``POST /runs/{run_id}/cancel`` — cancels a run this interface is publishing.

    A ticket is ``{"streamId", "token", "url", "expiresAt", "runId", "sessionId"}``.
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
        # Runs this interface is publishing: who started each, so that only they can cancel it.
        self._runs: dict[str, tuple[StreamHandle, str | None]] = {}

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
            handle = start_run(
                self.client,
                self.entity,
                body.message,
                session_id=session_id,
                user_id=user_id,
                ttl=self.ttl,
                protocol=self.protocol,
                publish_options=self.publish_options,
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
            requirements, tools = _requirements(body.tools)
            extra: dict[str, Any] = {} if tools is None else {"updated_tools": tools}
            handle = continue_run(
                self.client,
                entity,
                run_id=run_id,
                session_id=body.session_id,
                requirements=requirements,
                user_id=user_id,
                ttl=self.ttl,
                protocol=self.protocol,
                publish_options=self.publish_options,
                **extra,
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

        @router.post("/agui")
        async def agui(request: Request, run_input: RunAgentInput) -> dict[str, str | int]:  # pyright: ignore[reportUnusedFunction]
            props: object = run_input.forwarded_props
            client_user_id = cast("dict[str, object]", props).get("user_id") if isinstance(props, dict) else None
            user_id = _resolve_user(request, client_user_id if isinstance(client_user_id, str) else None)
            await _assert_session_writable(self.entity, run_input.thread_id, user_id, request)
            await self._check_agui_resume(run_input, request)
            handle = self._start_agui(run_input, user_id)
            self._keep(handle, user_id)
            return handle.to_dict()

    async def _check_agui_resume(self, run_input: Any, request: Request) -> None:
        """A trailing tool message resumes a paused run, which Agno's own AG-UI route refuses while an admin approval
        is pending; so does this one, where the installed Agno has the check."""
        try:
            from agno.os.interfaces.agui.input import extract_tool_messages
            from agno.os.interfaces.agui.resume import (  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
                find_resume_target_run_id,
            )
        except ImportError:  # agno 2.x has no such check
            return
        tool_messages = extract_tool_messages(run_input.messages or [])
        if not tool_messages:
            return
        run_id: object = await find_resume_target_run_id(self.entity, run_input.thread_id, tool_messages)
        reason = await _continuation_blocked(self.entity, run_id if isinstance(run_id, str) else None, request)
        if reason:
            raise HTTPException(status_code=403, detail=reason)

    def _start_agui(self, run_input: Any, user_id: str | None) -> StreamHandle:
        from agno.os.interfaces.agui.router import run_entity

        from ._agui import AguiFramer, AguiTrack
        from ._tee import Reporter

        options = self.publish_options
        report = Reporter(options.get("on_error"))
        ticket = self.client.ticket(None, self.ttl)
        publisher = self.client.apublisher(ticket.stream_id)

        def fits(data: str, event: str | None) -> bool:
            return publisher.frame_size(data, event) <= publisher.max_frame_bytes

        framer = AguiFramer(
            event_names=bool(options.get("event_names", False)),
            mask_errors=not options.get("expose_errors", False),
            shrink=options.get("on_oversize", "shrink") == "shrink",
            fits=fits,
            size=publisher.frame_size,
            limit=publisher.max_frame_bytes,
            report=report,
        )
        run_id: str = run_input.run_id
        track = AguiTrack(framer, run_id=run_id, thread_id=run_input.thread_id)
        run = _Run(publisher, track, options.get("cancelled", "fail"))
        entity: Any = self.entity
        before_close = options.get("before_close")
        task = asyncio.get_running_loop().create_task(
            publish_task(publisher, run, lambda: run_entity(entity, run_input, user_id=user_id), before_close),
            name=f"iq-stream-agno {run_id}",
        )
        _running.add(task)
        task.add_done_callback(_running.discard)
        return StreamHandle(ticket, run_id, run_input.thread_id, task)

    def _keep(self, handle: StreamHandle, user_id: str | None) -> None:
        self._runs[handle.run_id] = (handle, user_id)

        def forget(_: asyncio.Task[RunSummary]) -> None:
            if self._runs.get(handle.run_id, (None,))[0] is handle:
                del self._runs[handle.run_id]

        handle.task.add_done_callback(forget)

    def get_scope_mappings(self) -> dict[str, list[str]]:
        scope = [f"{self._family}:run"]
        routes = ["runs", "runs/*/continue", "runs/*/cancel"]
        if self.workflow is None and _has_agui():
            routes.append("agui")
        return {f"POST {self.prefix}/{route}": scope for route in routes}

    async def astart(self) -> None:
        return None

    async def aclose(self) -> None:
        """AgentOS's shutdown (agno 3): waits up to ``shutdown_timeout`` for the runs in flight, then cancels the
        rest, whose streams close with ``fail``. On agno 2.x, which has no such hook, call it from your lifespan."""
        tasks = [handle.task for handle, _ in self._runs.values()]
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


async def _continuation_blocked(entity: Any, run_id: str | None, request: Request) -> str | None:
    if run_id is None:
        return None
    from agno.os.auth import run_continuation_blocked_reason

    reason: object = await run_continuation_blocked_reason(
        getattr(entity, "db", None),
        run_id,
        authorization_enabled=bool(getattr(request.state, "authorization_enabled", False)),
        user_scopes=list(getattr(request.state, "scopes", None) or []),
        request=request,
    )
    return reason if isinstance(reason, str) and reason else None


def _requirements(tools: list[dict[str, Any]] | None) -> tuple[list[Any] | None, list[Any] | None]:
    """The paused tools as AgentOS's own continue route turns them into requirements."""
    if not tools:
        return None, None
    from agno.models.response import ToolExecution
    from agno.run.requirement import RunRequirement

    try:
        executions = [ToolExecution.from_dict(tool) for tool in tools]
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid structure or content for tools") from error
    return [RunRequirement(tool_execution=execution) for execution in executions], executions


async def _cancel(entity: Any, run_id: str) -> None:
    """Sets Agno's cancellation flag; the run then ends with ``RunCancelled`` and its stream with ``fail``."""
    await entity.acancel_run(run_id)
