"""The AgentOS interface's routes, in process: each answers with a ticket and a task publishes the run."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.os.middleware import user_scope
from agno.run.agent import CustomEvent, ReasoningContentDeltaEvent, ReasoningStartedEvent
from agno.tools import tool
from agno.workflow import Step, Workflow
from fastapi import FastAPI, Request, Response
from pydantic import TypeAdapter
from recorded import Errors, RecordingClient, names, payloads, types
from scripted_model import SLOW, ScriptedModel, weather_agent

from iq_stream_agno import MASKED_ERROR, RunSummary
from iq_stream_agno.os import IQStream

TICKET = {"streamId", "token", "url", "expiresAt", "runId", "sessionId"}


def agui_input(**changes: Any) -> dict[str, Any]:
    return {
        "threadId": "thread-1",
        "runId": "run-agui-1",
        "state": {},
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "messages": [{"id": "m1", "role": "user", "content": "Weather in Paris?"}],
        **changes,
    }


class Mounted:
    def __init__(self, interface: IQStream, authorization: bool = False) -> None:
        self.interface = interface
        self.app = FastAPI()
        self.app.include_router(interface.get_router())
        if authorization:
            # What AgentOS's authorization middleware leaves for the routes: on, and this caller has no scope.
            @self.app.middleware("http")
            async def authorize(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:  # pyright: ignore[reportUnusedFunction]
                request.state.authorization_enabled = True
                request.state.scopes = []
                return await call_next(request)

    async def post(self, path: str, body: Any = None, **params: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://app") as client:
            return await client.post(path, json=body, params=params)

    async def finished(self, run_id: str) -> RunSummary:
        handle, _ = self.interface._runs[run_id]  # pyright: ignore[reportPrivateUsage]
        return await asyncio.wait_for(handle.task, 10)


async def test_post_runs_answers_with_a_ticket_and_publishes_the_run() -> None:
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent()))
    response = await mounted.post("/iq/runs", {"message": "Weather in Paris?", "session_id": "s-1"})
    assert response.status_code == 200 and set(response.json()) == TICKET
    ticket = response.json()
    assert ticket["sessionId"] == "s-1" and client.verify("sub", ticket["token"], ticket["streamId"])
    summary = await mounted.finished(ticket["runId"])
    (publisher,) = client.recorders
    assert summary.outcome == publisher.outcome == "end" and names(publisher)[-1] == "RunCompleted"
    assert payloads(publisher)[0]["run_id"] == ticket["runId"]


async def test_post_agui_publishes_agno_s_own_ag_ui_conversion() -> None:
    from ag_ui.core import Event

    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent()))
    response = await mounted.post("/iq/agui", agui_input())
    assert response.status_code == 200 and set(response.json()) == TICKET
    ticket = response.json()
    assert (ticket["runId"], ticket["sessionId"]) == ("run-agui-1", "thread-1")
    await mounted.finished("run-agui-1")
    (publisher,) = client.recorders
    adapter: TypeAdapter[Any] = TypeAdapter(Event)
    for data in publisher.data:
        adapter.validate_json(data)
    first, last = payloads(publisher)[0], payloads(publisher)[-1]
    assert first == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-agui-1"}
    assert last["type"] == "RUN_FINISHED" and "TOOL_CALL_RESULT" in types(publisher)
    assert publisher.outcome == "end"


async def test_a_failing_ag_ui_run_ends_with_a_masked_run_error() -> None:
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(fail_on_turn=2)))
    await mounted.post("/iq/agui", agui_input(runId="run-agui-fail"))
    summary = await mounted.finished("run-agui-fail")
    (publisher,) = client.recorders
    last = payloads(publisher)[-1]
    assert last["type"] == "RUN_ERROR" and last["message"] == MASKED_ERROR
    assert "scripted model failure" not in "".join(publisher.data)
    assert summary.status == "error" and publisher.outcome == "fail"


async def test_only_who_started_a_run_can_cancel_it() -> None:
    SLOW["seconds"] = 1.0
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(tool_name="slow_weather")))
    assert (await mounted.post("/iq/runs/no-such-run/cancel")).status_code == 404
    ticket = (await mounted.post("/iq/runs", {"message": "Weather?", "user_id": "alice"})).json()
    (publisher,) = client.recorders
    for _ in range(500):
        if "ToolCallStarted" in names(publisher):
            break
        await asyncio.sleep(0.01)
    assert (await mounted.post(f"/iq/runs/{ticket['runId']}/cancel", user_id="bob")).status_code == 404
    cancelled = await mounted.post(f"/iq/runs/{ticket['runId']}/cancel", user_id="alice")
    assert cancelled.status_code == 200
    summary = await mounted.finished(ticket["runId"])
    assert summary.status == "cancelled" and "RunCancelled" in names(publisher) and publisher.outcome == "fail"


async def test_a_paused_run_continues_on_a_new_stream_for_its_own_user_only() -> None:
    client = RecordingClient()
    agent = weather_agent(tool_name="book_trip", answer="Your trip is booked.", db=InMemoryDb())
    mounted = Mounted(IQStream(client, agent=agent))
    first = (
        await mounted.post("/iq/runs", {"message": "Book Paris", "session_id": "s-hitl-route", "user_id": "ann"})
    ).json()
    await mounted.finished(first["runId"])
    paused = payloads(client.recorders[0])[-1]
    assert paused["event"] == "RunPaused"
    tools = [{**tool, "confirmed": True} for tool in paused["tools"]]
    body = {"session_id": "s-hitl-route", "tools": tools, "user_id": "ann"}

    assert (await mounted.post("/iq/runs/no-such-run/continue", body)).status_code == 404
    assert (await mounted.post(f"/iq/runs/{first['runId']}/continue", {**body, "user_id": "eve"})).status_code == 404
    response = await mounted.post(f"/iq/runs/{first['runId']}/continue", body)
    assert response.status_code == 200
    second = response.json()
    assert second["runId"] == first["runId"] and second["streamId"] != first["streamId"]
    summary = await mounted.finished(first["runId"])
    publisher = client.recorders[1]
    assert names(publisher)[0] == "RunContinued" and "booked Paris" in "".join(publisher.data)
    assert summary.status == "completed" and publisher.outcome == "end"


async def test_shutdown_cancels_the_runs_still_in_flight() -> None:
    SLOW["seconds"] = 2.0
    client = RecordingClient()
    interface = IQStream(client, agent=weather_agent(tool_name="slow_weather"), shutdown_timeout=0.2)
    mounted = Mounted(interface)
    ticket = (await mounted.post("/iq/runs", {"message": "Weather?"})).json()
    handle, _ = interface._runs[ticket["runId"]]  # pyright: ignore[reportPrivateUsage]
    await interface.aclose()
    assert handle.task.done() and client.recorders[0].outcome == "fail"


def test_scopes_and_routes_follow_the_entity() -> None:
    agent_interface = IQStream(RecordingClient(), agent=weather_agent(), prefix="/stream")
    assert agent_interface.get_scope_mappings() == {
        "POST /stream/runs": ["agents:run"],
        "POST /stream/runs/*/continue": ["agents:run"],
        "POST /stream/runs/*/cancel": ["agents:run"],
        "POST /stream/agui": ["agents:run"],
    }
    workflow = Workflow(id="wf", name="WF", steps=[Step(name="lookup", agent=weather_agent())], telemetry=False)
    workflow_interface = IQStream(RecordingClient(), workflow=workflow)
    mappings = workflow_interface.get_scope_mappings()
    assert "POST /iq/agui" not in mappings and all(scope == ["workflows:run"] for scope in mappings.values())
    paths = {route.path for route in workflow_interface.get_router().routes}  # type: ignore[attr-defined]
    assert "/iq/agui" not in paths and "/iq/runs" in paths
    with pytest.raises(ValueError):
        IQStream(RecordingClient())


async def test_a_workflow_run_is_published_through_post_runs() -> None:
    client = RecordingClient()
    workflow = Workflow(id="wf", name="WF", steps=[Step(name="lookup", agent=weather_agent())], telemetry=False)
    mounted = Mounted(IQStream(client, workflow=workflow))
    ticket = (await mounted.post("/iq/runs", {"message": "Weather?"})).json()
    summary = await mounted.finished(ticket["runId"])
    (publisher,) = client.recorders
    assert names(publisher)[0] == "WorkflowStarted" and names(publisher)[-1] == "WorkflowCompleted"
    assert summary.run_id == ticket["runId"] and publisher.outcome == "end"


async def until(condition: Callable[[], bool]) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never held")


def raw_names(publisher: Any) -> list[str]:
    return [row["event"].get("event") for row in payloads(publisher) if row["type"] == "RAW"]


async def test_post_agui_masks_a_failing_tool_s_error_text() -> None:
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(tool_name="broken_tool")))
    await mounted.post("/iq/agui", agui_input(runId="run-agui-broken"))
    await mounted.finished("run-agui-broken")
    (publisher,) = client.recorders
    assert "secret database password" not in "".join(publisher.data)
    (result,) = [row for row in payloads(publisher) if row["type"] == "TOOL_CALL_RESULT"]
    assert MASKED_ERROR in result["content"] and "ToolCallError" in raw_names(publisher)


def resume_input(run_id: str, call: dict[str, Any], answer: str = '{"accepted": true}') -> dict[str, Any]:
    tool_call = {"id": call["toolCallId"], "type": "function", "function": {"name": call["toolCallName"]}}
    tool_call["function"]["arguments"] = '{"city": "Paris"}'
    return agui_input(
        runId=run_id,
        messages=[
            {"id": "m1", "role": "user", "content": "Book Paris"},
            {"id": "a1", "role": "assistant", "content": "", "toolCalls": [tool_call]},
            {"id": "t1", "role": "tool", "toolCallId": call["toolCallId"], "content": answer},
        ],
    )


async def paused_over_agui(mounted: Mounted, client: RecordingClient, thread: str) -> tuple[str, dict[str, Any]]:
    first = (await mounted.post("/iq/agui", agui_input(threadId=thread, runId=f"{thread}-leg-1"))).json()
    summary = await mounted.finished(first["runId"])
    assert summary.status == "paused" and client.recorders[0].outcome == "end"
    call = next(row for row in payloads(client.recorders[0]) if row["type"] == "TOOL_CALL_START")
    return first["runId"], call


async def test_an_ag_ui_resume_is_published_and_cancelled_under_the_paused_run_s_id() -> None:
    SLOW["seconds"] = 1.0
    client = RecordingClient()
    agent = weather_agent(tool_name="slow_book", answer="Your trip is booked.", db=InMemoryDb())
    mounted = Mounted(IQStream(client, agent=agent))
    paused_run_id, call = await paused_over_agui(mounted, client, "t-resume")

    response = await mounted.post("/iq/agui", resume_input("t-resume-leg-2", call) | {"threadId": "t-resume"})
    assert response.status_code == 200
    ticket = response.json()
    assert (ticket["runId"], ticket["sessionId"]) == (paused_run_id, "t-resume")
    publisher = client.recorders[1]
    await until(lambda: "ToolCallStarted" in raw_names(publisher) or "TOOL_CALL_START" in types(publisher))
    assert (await mounted.post(f"/iq/runs/{ticket['runId']}/cancel")).status_code == 200
    summary = await mounted.finished(paused_run_id)
    rows = payloads(publisher)
    # The AG-UI run is the client's; the Agno run, its events, the ticket and the cancel are the paused run's.
    assert rows[0] == {"type": "RUN_STARTED", "threadId": "t-resume", "runId": "t-resume-leg-2"}
    assert {row["event"]["run_id"] for row in rows if row["type"] == "RAW"} == {paused_run_id}
    assert rows[-1]["type"] == "RUN_FINISHED" and rows[-1]["outcome"] == {"type": "cancelled"}
    assert summary.status == "cancelled" and publisher.outcome == "fail"


async def test_an_ag_ui_resume_that_answers_no_paused_run_is_refused_before_a_ticket() -> None:
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(tool_name="book_trip", db=InMemoryDb())))
    call = {"toolCallId": "call_unknown", "toolCallName": "book_trip"}
    response = await mounted.post("/iq/agui", resume_input("run-orphan", call) | {"threadId": "t-orphan"})
    assert response.status_code == 400 and client.recorders == []


async def test_a_run_id_in_flight_is_refused_a_second_time() -> None:
    SLOW["seconds"] = 1.0
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(tool_name="slow_weather")))
    assert (await mounted.post("/iq/agui", agui_input(runId="run-twice"))).status_code == 200
    assert (await mounted.post("/iq/agui", agui_input(runId="run-twice"))).status_code == 409
    assert len(client.recorders) == 1
    await mounted.finished("run-twice")


async def test_post_agui_cancel_finishes_with_the_cancelled_outcome() -> None:
    SLOW["seconds"] = 1.0
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(tool_name="slow_weather")))
    await mounted.post("/iq/agui", agui_input(runId="run-agui-cancel"))
    (publisher,) = client.recorders
    await until(lambda: "TOOL_CALL_START" in types(publisher))
    assert (await mounted.post("/iq/runs/run-agui-cancel/cancel")).status_code == 200
    summary = await mounted.finished("run-agui-cancel")
    last = payloads(publisher)[-1]
    assert last["type"] == "RUN_FINISHED" and last["outcome"] == {"type": "cancelled"}
    assert summary.status == "cancelled" and publisher.outcome == "fail"


class Thinking(Agent):
    """An agent that opens a reasoning span and then thinks for a long time."""

    def arun(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        run_id, session_id = kwargs["run_id"], kwargs["session_id"]

        async def events() -> AsyncIterator[Any]:
            yield ReasoningStartedEvent(run_id=run_id, session_id=session_id)
            yield ReasoningContentDeltaEvent(run_id=run_id, session_id=session_id, reasoning_content="Hmm")
            await asyncio.sleep(10)

        return events()


async def test_a_shutdown_closes_what_is_open_and_finishes_the_ag_ui_run_as_cancelled() -> None:
    client = RecordingClient()
    interface = IQStream(client, agent=Thinking(id="thinker", model=None, telemetry=False), shutdown_timeout=0.1)
    mounted = Mounted(interface)
    await mounted.post("/iq/agui", agui_input(runId="run-thinking"))
    (publisher,) = client.recorders
    await until(lambda: "REASONING_MESSAGE_CONTENT" in types(publisher))
    await interface.aclose()
    kinds = [kind for kind in types(publisher) if kind != "STATE_SNAPSHOT"]  # the request's state, as Agno sends it
    assert kinds[-3:] == ["REASONING_MESSAGE_END", "REASONING_END", "RUN_FINISHED"]
    assert payloads(publisher)[-1]["outcome"] == {"type": "cancelled"} and publisher.outcome == "fail"


@dataclass
class Progress(CustomEvent):
    event: str = "Progress"
    handle: Any = field(default_factory=object)  # an application object, which JSON cannot encode


@tool
def lookup(city: str) -> Iterator[Any]:
    """Look a city up.

    Args:
        city (str): City name
    """
    yield Progress()
    yield f"21C in {city}"


async def test_post_agui_skips_an_event_it_cannot_encode_and_the_run_completes() -> None:
    client = RecordingClient()
    errors = Errors()
    agent = Agent(model=ScriptedModel(tool_name="lookup"), tools=[lookup], db=InMemoryDb(), telemetry=False)
    mounted = Mounted(IQStream(client, agent=agent, publish_options={"on_error": errors}))
    await mounted.post("/iq/agui", agui_input(runId="run-agui-progress"))
    summary = await mounted.finished("run-agui-progress")
    (publisher,) = client.recorders
    assert [error.code for error in errors] == ["encode_failed"] and types(publisher)[-1] == "RUN_FINISHED"
    assert summary.status == "completed" and publisher.outcome == "end"
    stored: Any = await agent.aget_run_output("run-agui-progress", session_id="thread-1")
    assert str(getattr(stored.status, "value", stored.status)).lower() == "completed"


@pytest.mark.skipif(
    not hasattr(user_scope, "assert_session_writable"),
    reason="agno 2.x checks no session's owner, on its routes either",
)
async def test_another_user_s_session_is_refused_on_both_start_routes() -> None:
    client = RecordingClient()
    mounted = Mounted(IQStream(client, agent=weather_agent(db=InMemoryDb())))
    first = await mounted.post("/iq/runs", {"message": "Weather?", "session_id": "s-alice", "user_id": "alice"})
    await mounted.finished(first.json()["runId"])
    runs = await mounted.post("/iq/runs", {"message": "Weather?", "session_id": "s-alice", "user_id": "bob"})
    agui = await mounted.post("/iq/agui", agui_input(threadId="s-alice", forwardedProps={"user_id": "bob"}))
    assert runs.status_code == agui.status_code == 404 and len(client.recorders) == 1


class ApprovalsDb(InMemoryDb):
    """A database with a pending admin approval for every run it is asked about."""

    def get_approvals(self, **query: Any) -> list[dict[str, Any]]:
        return [{"id": "approval-1", **query}]


async def test_a_run_waiting_for_an_admin_approval_is_not_continued_on_either_route() -> None:
    client = RecordingClient()
    agent = weather_agent(tool_name="book_trip", db=ApprovalsDb())
    mounted = Mounted(IQStream(client, agent=agent), authorization=True)
    paused_run_id, call = await paused_over_agui(mounted, client, "t-approval")
    tools = [{**tool, "confirmed": True} for tool in payloads_of_paused(agent, paused_run_id, "t-approval")]
    body = {"session_id": "t-approval", "tools": tools}
    continued = await mounted.post(f"/iq/runs/{paused_run_id}/continue", body)
    resumed = await mounted.post("/iq/agui", resume_input("t-approval-leg-2", call) | {"threadId": "t-approval"})
    for refused in (continued, resumed):
        assert refused.status_code == 403 and "admin approval" in refused.json()["detail"]
    assert len(client.recorders) == 1


def payloads_of_paused(agent: Agent, run_id: str, session_id: str) -> list[dict[str, Any]]:
    stored: Any = agent.get_run_output(run_id, session_id=session_id)
    return [json.loads(json.dumps(tool.to_dict())) for tool in stored.tools]


def test_bad_publish_options_are_refused_when_the_interface_is_made() -> None:
    with pytest.raises(ValueError, match="cancelled"):
        IQStream(RecordingClient(), agent=weather_agent(), publish_options={"cancelled": "End"})  # type: ignore[typeddict-item]


async def test_post_agui_leaves_out_the_excluded_agno_events() -> None:
    client = RecordingClient()
    options: Any = {"exclude": ("ModelRequestStarted", "ModelRequestCompleted")}
    mounted = Mounted(IQStream(client, agent=weather_agent(), publish_options=options))
    await mounted.post("/iq/agui", agui_input(runId="run-agui-exclude"))
    await mounted.finished("run-agui-exclude")
    (publisher,) = client.recorders
    assert "ModelRequestStarted" not in raw_names(publisher) and "RunStarted" in raw_names(publisher)
