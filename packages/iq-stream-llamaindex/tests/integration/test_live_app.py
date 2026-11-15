"""The FastAPI example (examples/fastapi_agui/app.py) served by uvicorn on a free port and driven over HTTP: start a
run, subscribe with the ticket it answered, answer a human-input request, cancel a run."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from checks import agui_events, assert_well_formed, joined
from fakes import ANSWER, FIRST_TEXT, hitl_agent, slow_weather, weather_agent
from iq_stream import IQClient
from iq_stream.testing import TestServers, free_port, subscribe, subscribe_in_background
from live import assert_delivered
from workflows import Workflow

from iq_stream_llamaindex import RunOutcome

APP = Path(__file__).resolve().parents[2] / "examples" / "fastapi_agui" / "app.py"


def load_example() -> Any:
    spec = importlib.util.spec_from_file_location("fastapi_agui_app", APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


@contextmanager
def served(agent: Workflow, iq: IQClient, answers: dict[str, str]) -> Iterator[str]:
    """The example app with this agent, under uvicorn in a thread of its own, stopped on the way out."""

    async def save_answer(thread_id: str, outcome: RunOutcome) -> None:
        answers[thread_id] = str(outcome.result) if outcome.status == "completed" else outcome.status

    app = load_example().create_app(agent, iq, save_answer)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline and thread.is_alive(), "uvicorn did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def post(url: str, body: dict[str, Any]) -> httpx.Response:
    return httpx.post(url, json=body, timeout=10, trust_env=False)


def test_a_chat_request_answers_a_ticket_and_the_run_reaches_its_subscriber(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    answers: dict[str, str] = {}
    with served(weather_agent(), iq_client, answers) as base:
        response = post(f"{base}/chat", {"thread_id": "thread-7", "message": "Weather in Paris?"})
        assert response.status_code == 200
        ticket = response.json()
        assert set(ticket) == {"streamId", "token", "url", "expiresAt"}
        received = subscribe(ticket["url"], timeout=15)  # the ticket's complete URL, as EventSource opens it
        assert_delivered(received, "iq:end")
        events = agui_events(received.data)
        assert_well_formed(events)
        assert events[0]["threadId"] == "thread-7" and events[0]["runId"] == ticket["streamId"]
        assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT + ANSWER
        assert answers == {"thread-7": ANSWER}  # saved before the close line


def test_a_human_input_request_is_answered_through_the_app_and_the_same_stream_goes_on(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    answers: dict[str, str] = {}
    with served(hitl_agent(), iq_client, answers) as base:
        ticket = post(f"{base}/chat", {"thread_id": "thread-8", "message": "Do it"}).json()
        subscription = subscribe_in_background(ticket["url"])
        answer_url = f"{base}/chat/{ticket['streamId']}/answer"
        assert post(answer_url, {"token": "forged", "response": "yes"}).status_code == 403
        deadline = time.monotonic() + 10
        while (status := post(answer_url, {"token": ticket["token"], "response": "yes"}).status_code) == 409:
            assert time.monotonic() < deadline, "the question never came"
            time.sleep(0.05)
        assert status == 204
        received = subscription.result(15)
        assert_delivered(received, "iq:end")
        events = agui_events(received.data)
        assert_well_formed(events)
        question = next(event for event in events if event["type"] == "CUSTOM")
        assert question == {"type": "CUSTOM", "name": "input_required", "value": {"prefix": "Proceed?"}}
        result = next(event for event in events if event["type"] == "TOOL_CALL_RESULT")
        assert result["content"] == "yes" and joined(events, "TEXT_MESSAGE_CONTENT") == "Done."
        assert answers == {"thread-8": "Done."}


def test_a_run_cancelled_through_the_app_ends_with_the_cancelled_outcome_and_iq_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    answers: dict[str, str] = {}
    with served(weather_agent(tool=slow_weather(5.0)), iq_client, answers) as base:
        ticket = post(f"{base}/chat", {"thread_id": "thread-9", "message": "Weather?"}).json()
        subscription = subscribe_in_background(ticket["url"])
        response = post(f"{base}/chat/{ticket['streamId']}/cancel", {"token": ticket["token"]})
        assert response.status_code == 204
        received = subscription.result(15)
        assert_delivered(received, "iq:fail")
        events = agui_events(received.data)
        assert events[-1]["type"] == "RUN_FINISHED" and events[-1]["outcome"] == {"type": "cancelled"}
        assert answers == {"thread-9": "cancelled"}
        unknown = post(f"{base}/chat/{ticket['streamId']}/cancel", {"token": ticket["token"]})
        assert unknown.status_code == 404  # the run is over and forgotten
