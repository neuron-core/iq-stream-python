"""The README's FastAPI route, served by uvicorn on a free port and driven over HTTP as StreamAgent would."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import uvicorn
from agui_checks import checked, conversation
from crew_fakes import ANSWER, make_crew
from iq_stream.testing import TestServers, free_port, subscribe, subscribe_in_background
from readme_blocks import ROUTE


@pytest.fixture
def chat_app(iq_servers: TestServers, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, dict[str, Any]]]:
    monkeypatch.setenv("IQ_STREAM_URL", iq_servers.default.url)
    monkeypatch.setenv("IQ_STREAM_APP_KEY", iq_servers.app_key)
    monkeypatch.setenv("IQ_STREAM_APP_SECRET", iq_servers.app_secret)
    namespace: dict[str, Any] = {"build_crew": lambda: make_crew(delay=0.005)}
    exec(compile(ROUTE, "README.md", "exec"), namespace)
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(namespace["app"], host="127.0.0.1", port=port, log_level="warning", lifespan="off", ws="none")
    )
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}", namespace
    finally:
        server.should_exit = True
        thread.join(10)
        namespace["workers"].shutdown(wait=True)


def run_input(thread_id: str, run_id: str) -> dict[str, Any]:
    return {
        "threadId": thread_id,
        "runId": run_id,
        "state": {},
        "messages": [{"id": "u1", "role": "user", "content": "What is the capital of France?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def test_a_chat_over_http(chat_app: tuple[str, dict[str, Any]], iq_servers: TestServers) -> None:
    url, namespace = chat_app
    with httpx.Client(base_url=url, trust_env=False, timeout=10) as http:
        ticket = http.post("/chat", json=run_input("thread-1", "run-1")).raise_for_status().json()
        assert set(ticket) == {"streamId", "token", "url", "expiresAt"}
        received = subscribe(ticket["url"], timeout=30)  # the ticket's URL, as the browser opens it
        assert received.terminal is not None and received.terminal.event == "iq:end"
        events = checked(received.data)
        assert events[0] == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-1"}
        assert conversation(events).text == ANSWER
        assert [frame.id for frame in received.frames] == [str(n) for n in range(1, len(received.frames) + 1)]
        # Saved before the close line: a browser that arrives late is told so, and loads the answer.
        assert http.get(f"/chat/{ticket['streamId']}").json() == {"answer": ANSWER}
        late = subscribe(iq_servers.default.url, ticket["streamId"], ticket["token"], timeout=10)
        assert late.frames == []
        assert late.terminal is not None and late.terminal.event == "iq:gone"
    assert namespace["answers"] == {ticket["streamId"]: ANSWER}


def test_two_chats_at_once(chat_app: tuple[str, dict[str, Any]], iq_servers: TestServers) -> None:
    url, _ = chat_app
    with httpx.Client(base_url=url, trust_env=False, timeout=10) as http:
        tickets = [http.post("/chat", json=run_input(f"thread-{n}", f"run-{n}")).json() for n in range(2)]
    subscriptions = [subscribe_in_background(ticket["url"], wait_open=None) for ticket in tickets]
    for number, subscription in enumerate(subscriptions):
        received = subscription.result(30)
        assert received.terminal is not None and received.terminal.event == "iq:end"
        events = checked(received.data)
        assert events[0]["threadId"] == f"thread-{number}"
        assert conversation(events).text == ANSWER
