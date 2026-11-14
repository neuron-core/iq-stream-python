"""The AgentOS interface's routes, in process: each answers with a ticket and a task publishes the run."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from agno.db.in_memory import InMemoryDb
from agno.workflow import Step, Workflow
from fastapi import FastAPI
from pydantic import TypeAdapter
from recorded import RecordingClient, names, payloads, types
from scripted_model import SLOW, weather_agent

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
    def __init__(self, interface: IQStream) -> None:
        self.interface = interface
        self.app = FastAPI()
        self.app.include_router(interface.get_router())

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
    assert last["type"] == "RUN_ERROR" and last["message"] == MASKED_ERROR and "rawEvent" not in last
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
