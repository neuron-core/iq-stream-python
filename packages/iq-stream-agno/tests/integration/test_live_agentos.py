"""The AgentOS interface in AgentOS(...).get_app() under uvicorn, driven over HTTP, publishing to the real server."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn
from ag_ui.core import Event
from agno.db.in_memory import InMemoryDb
from agno.os import AgentOS
from agno.os.interfaces.base import BaseInterface
from iq_stream import IQClient
from iq_stream.testing import TestServers, free_port, subscribe, subscribe_in_background
from live import delivered
from pydantic import TypeAdapter
from scripted_model import SLOW, weather_agent

from iq_stream_agno.os import IQStream


class Served:
    """An AgentOS app on a free port, in a thread, until stopped."""

    def __init__(self, interface: IQStream) -> None:
        agent = interface.entity
        app = AgentOS(agents=[agent], interfaces=[interface], telemetry=False).get_app()  # type: ignore[list-item]
        self.url = f"http://127.0.0.1:{free_port()}"
        port = int(self.url.rsplit(":", 1)[1])
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, name="agentos", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            assert time.monotonic() < deadline, "AgentOS did not start"
            time.sleep(0.02)

    def post(self, path: str, body: Any = None, **params: str) -> tuple[httpx.Response, float]:
        started = time.monotonic()
        response = httpx.post(f"{self.url}{path}", json=body, params=params, timeout=10, trust_env=False)
        return response, time.monotonic() - started

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(20)
        assert not self.thread.is_alive(), "AgentOS did not stop"


@pytest.fixture
def served(iq_servers: TestServers) -> Iterator[Served]:
    client = IQClient(iq_servers.default.url, iq_servers.app_key, iq_servers.app_secret, min_open=0.0)
    interface = IQStream(client, agent=weather_agent(tool_name="slow_weather"), shutdown_timeout=0.5)
    server = Served(interface)
    try:
        yield server
    finally:
        server.stop()


def test_post_runs_answers_at_once_and_the_subscriber_gets_the_run(served: Served) -> None:
    SLOW["seconds"] = 0.3
    response, took = served.post("/iq/runs", {"message": "Weather in Paris?"})
    assert response.status_code == 200 and took < 1.0
    ticket = response.json()
    rows = delivered(subscribe(ticket["url"], timeout=15), "iq:end")
    assert rows[0]["event"] == "RunStarted" and rows[0]["run_id"] == ticket["runId"]
    assert rows[-1]["event"] == "RunCompleted"


def test_post_agui_answers_at_once_and_the_subscriber_gets_ag_ui_events(served: Served) -> None:
    SLOW["seconds"] = 0.3
    body = {
        "threadId": "thread-1",
        "runId": "run-agui-live",
        "state": {},
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "messages": [{"id": "m1", "role": "user", "content": "Weather in Paris?"}],
    }
    response, took = served.post("/iq/agui", body)
    assert response.status_code == 200 and took < 1.0
    subscription = subscribe(response.json()["url"], timeout=15)
    rows = delivered(subscription, "iq:end")
    adapter: TypeAdapter[Any] = TypeAdapter(Event)
    for frame in subscription.frames:
        adapter.validate_json(frame.data)
    assert rows[0] == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-agui-live"}
    assert rows[-1]["type"] == "RUN_FINISHED" and "TOOL_CALL_RESULT" in [row["type"] for row in rows]


def agui_body(run_id: str, thread_id: str = "thread-1", messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "threadId": thread_id,
        "runId": run_id,
        "state": {},
        "tools": [],
        "context": [],
        "forwardedProps": {},
        "messages": messages or [{"id": "m1", "role": "user", "content": "Weather in Paris?"}],
    }


def test_an_ag_ui_run_cancelled_through_the_cancel_route_finishes_as_cancelled(served: Served) -> None:
    SLOW["seconds"] = 2.0
    ticket = served.post("/iq/agui", agui_body("run-agui-cancel-live"))[0].json()
    subscription = subscribe_in_background(ticket["url"], wait_open=5.0)
    time.sleep(0.5)  # the tool is running
    assert served.post(f"/iq/runs/{ticket['runId']}/cancel")[0].status_code == 200
    last = delivered(subscription.result(15), "iq:fail")[-1]
    assert last["type"] == "RUN_FINISHED" and last["outcome"] == {"type": "cancelled"}


def test_an_ag_ui_resume_is_cancelled_by_the_run_id_of_its_ticket(iq_servers: TestServers) -> None:
    SLOW["seconds"] = 2.0
    client = IQClient(iq_servers.default.url, iq_servers.app_key, iq_servers.app_secret, min_open=0.0)
    agent = weather_agent(tool_name="slow_book", answer="Your trip is booked.", db=InMemoryDb())
    server = Served(IQStream(client, agent=agent, shutdown_timeout=0.5))
    try:
        user = {"id": "m1", "role": "user", "content": "Book Paris"}
        first = server.post("/iq/agui", agui_body("live-leg-1", "t-live-resume", [user]))[0].json()
        call = next(
            row for row in delivered(subscribe(first["url"], timeout=15), "iq:end") if row["type"] == "TOOL_CALL_START"
        )
        tool_call = {"id": call["toolCallId"], "type": "function", "function": {"name": "slow_book", "arguments": "{}"}}
        answer = {"id": "t1", "role": "tool", "toolCallId": call["toolCallId"], "content": '{"accepted": true}'}
        assistant = {"id": "a1", "role": "assistant", "content": "", "toolCalls": [tool_call]}
        body = agui_body("live-leg-2", "t-live-resume", [user, assistant, answer])
        second = server.post("/iq/agui", body)[0].json()
        assert second["runId"] == "live-leg-1" and second["streamId"] != first["streamId"]
        subscription = subscribe_in_background(second["url"], wait_open=5.0)
        time.sleep(0.5)  # the confirmed tool is running
        assert server.post(f"/iq/runs/{second['runId']}/cancel")[0].status_code == 200
        frames = delivered(subscription.result(15), "iq:fail")
        assert frames[0] == {"type": "RUN_STARTED", "threadId": "t-live-resume", "runId": "live-leg-2"}
        assert frames[-1]["type"] == "RUN_FINISHED" and frames[-1]["outcome"] == {"type": "cancelled"}
    finally:
        server.stop()


def test_the_cancel_route_ends_the_stream_with_iq_fail(served: Served) -> None:
    SLOW["seconds"] = 2.0
    ticket = served.post("/iq/runs", {"message": "Weather in Paris?"})[0].json()
    subscription = subscribe_in_background(ticket["url"], wait_open=5.0)
    time.sleep(0.5)  # the tool is running
    response, _ = served.post(f"/iq/runs/{ticket['runId']}/cancel")
    assert response.status_code == 200
    rows = delivered(subscription.result(15), "iq:fail")
    assert "RunCancelled" in [row["event"] for row in rows]


@pytest.mark.skipif(not hasattr(BaseInterface, "aclose"), reason="agno 2.x calls no interface on shutdown")
def test_a_shutdown_with_a_run_in_flight_closes_its_stream(served: Served) -> None:
    SLOW["seconds"] = 5.0
    ticket = served.post("/iq/runs", {"message": "Weather in Paris?"})[0].json()
    subscription = subscribe_in_background(ticket["url"], wait_open=5.0)
    time.sleep(0.5)
    served.stop()
    result = subscription.result(15)
    assert result.terminal is not None and result.terminal.event == "iq:fail"
    assert result.terminal.data["reason"] == "publisher"
