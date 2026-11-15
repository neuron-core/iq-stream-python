"""Every Python block of the README is run here, against the default test server.

Scripts run in a process of their own after a prelude that defines what the README says is the application's
(``model``, ``get_weather``, ``save_answer()``, an ``agent``, its ``config`` and a ``question``), with the credentials
in the environment; they must exit cleanly and print nothing to stderr, where a publisher that gave up would have
logged its warning. Blocks that make a FastAPI app are served by uvicorn on a free port and used over HTTP, as a
page would: start a run, read it, load the thread, stop a run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fakes import FINAL, FIRST, text_of
from iq_stream.testing import TestServers, free_port, subscribe

from iq_stream_langchain import CANCELLED_PART

TESTS = Path(__file__).resolve().parents[1]
README = TESTS.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)
APPS = [block for block in BLOCKS if "FastAPI()" in block]
SCRIPTS = [block for block in BLOCKS if block not in APPS]

PRELUDE = """
import warnings

from fakes import booking_agent, get_weather, scripted_model
from langchain.agents import create_agent
from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.memory import InMemorySaver

warnings.filterwarnings("ignore", category=LangChainBetaWarning)  # stream_events(version="v3") is beta
model = scripted_model(repeat=True, token_delay={delay})
saved = []

def save_answer(messages):
    saved.append(messages)

agent = {agent}
config = {{"configurable": {{"thread_id": "readme"}}}}
question = {{"messages": [{{"role": "user", "content": "What is the weather in Turin?"}}]}}
"""

WEATHER = "create_agent(model, [get_weather], checkpointer=InMemorySaver())"
BOOKING = "booking_agent(checkpointer=InMemorySaver())"  # its tool waits for a human

EPILOGUE = """
from iq_stream import AsyncPublisher as _AsyncPublisher, Publisher as _Publisher
for _value in list(globals().values()):
    if isinstance(_value, (_Publisher, _AsyncPublisher)):
        assert not _value.alive and _value.error is None, _value
"""


def environment(servers: TestServers) -> dict[str, str]:
    return {
        "IQ_STREAM_URL": servers.default.url,
        "IQ_STREAM_PUBLIC_URL": servers.default.url,
        "IQ_STREAM_APP_KEY": servers.app_key,
        "IQ_STREAM_APP_SECRET": servers.app_secret,
    }


def test_the_readme_has_samples_of_each_kind() -> None:
    assert len(SCRIPTS) == 5
    assert len(APPS) == 2


@pytest.mark.parametrize("block", SCRIPTS, ids=[f"script-{n}" for n in range(len(SCRIPTS))])
def test_a_readme_script_runs(block: str, iq_servers: TestServers) -> None:
    process = subprocess.run(
        [sys.executable, "-c", PRELUDE.format(delay=0.0, agent=WEATHER) + block + EPILOGUE],
        env={**os.environ, **environment(iq_servers), "PYTHONPATH": str(TESTS)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert process.stderr == ""


class Served:
    """A README app served by uvicorn in a thread, on a free port."""

    def __init__(
        self, block: str, monkeypatch: pytest.MonkeyPatch, servers: TestServers, delay: float, agent: str = WEATHER
    ) -> None:
        for name, value in environment(servers).items():
            monkeypatch.setenv(name, value)
        # A module of its own, as an app is: pydantic resolves the annotations of the app's models in it.
        module = types.ModuleType("readme_app")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(PRELUDE.format(delay=delay, agent=agent) + block, module.__dict__)  # the README's own code
        self.namespace = module.__dict__
        port = free_port()
        self.url = f"http://127.0.0.1:{port}"
        app = self.namespace["app"]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(10)


@pytest.fixture
def chat_app(monkeypatch: pytest.MonkeyPatch, iq_servers: TestServers) -> Iterator[Served]:
    served = Served(next(block for block in APPS if "/chat" in block), monkeypatch, iq_servers, delay=0.02)
    yield served
    served.stop()


@pytest.fixture
def booking_app(monkeypatch: pytest.MonkeyPatch, iq_servers: TestServers) -> Iterator[Served]:
    served = Served(next(block for block in APPS if "/chat" in block), monkeypatch, iq_servers, 0.0, BOOKING)
    yield served
    served.stop()


@pytest.fixture
def agent_app(monkeypatch: pytest.MonkeyPatch, iq_servers: TestServers) -> Iterator[Served]:
    served = Served(next(block for block in APPS if "/agent" in block), monkeypatch, iq_servers, delay=0.02)
    yield served
    served.stop()


def test_the_chat_route_publishes_a_run_and_the_thread_can_be_loaded(chat_app: Served, iq_servers: TestServers) -> None:
    with httpx.Client(base_url=chat_app.url, trust_env=False, timeout=10) as http:
        ticket = http.post("/chat", json={"thread_id": "t-1", "message": "What is the weather in Turin?"}).json()
        assert set(ticket) == {"streamId", "token", "url", "expiresAt"}
        received = subscribe(ticket["url"], timeout=20)  # after the answer: the route answered at once
        assert received.terminal is not None and received.terminal.event == "iq:end"
        parts = [json.loads(data) for data in received.data]
        assert parts[0] == {"type": "metadata", "ns": [], "data": {"run_id": None, "thread_id": "t-1"}}
        assert text_of(parts) == FIRST + FINAL

        late = subscribe(ticket["url"], timeout=10)
        assert late.terminal is not None and late.terminal.event == "iq:gone"
        history = http.get("/chat/t-1").json()
        assert history[-1] == {"type": "ai", "content": FINAL}


def test_the_stop_route_cancels_a_run(chat_app: Served, iq_servers: TestServers) -> None:
    with httpx.Client(base_url=chat_app.url, trust_env=False, timeout=10) as http:
        ticket = http.post("/chat", json={"thread_id": "t-2", "message": "What is the weather in Turin?"}).json()
        time.sleep(0.2)
        other = http.post("/chat", json={"thread_id": "t-other", "message": "What is the weather in Turin?"}).json()
        refused = http.post(f"/chat/{ticket['streamId']}/stop", json={"token": other["token"]})
        assert refused.status_code == 403, "another stream's token"
        assert http.post(f"/chat/{ticket['streamId']}/stop", json={"token": ticket["token"]}).status_code == 204
        received = subscribe(ticket["url"], timeout=20)
        went_on = subscribe(other["url"], timeout=20)
        assert went_on.terminal is not None and went_on.terminal.event == "iq:end", "the other run went on"
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.data[-1] == CANCELLED_PART
    deadline = time.monotonic() + 5  # the task ends once the server has answered the close line
    while chat_app.namespace["running"]:
        assert time.monotonic() < deadline, "the cancelled task did not end"
        time.sleep(0.02)


def test_a_reloaded_page_gets_the_run_in_flight_and_its_stream_from_the_first_frame(
    chat_app: Served, iq_servers: TestServers
) -> None:
    with httpx.Client(base_url=chat_app.url, trust_env=False, timeout=10) as http:
        assert http.get("/chat/t-4/stream").status_code == 204, "nothing in flight yet"
        started = http.post("/chat", json={"thread_id": "t-4", "message": "What is the weather in Turin?"}).json()
        time.sleep(0.3)  # the page reloads while the answer is written
        ticket = http.get("/chat/t-4/stream").json()
        assert ticket["streamId"] == started["streamId"]
        received = subscribe(ticket["url"], timeout=20)
        assert received.terminal is not None and received.terminal.event == "iq:end"
        parts = [json.loads(data) for data in received.data]
        assert parts[0]["type"] == "metadata" and text_of(parts) == FIRST + FINAL, "replayed from the first frame"
        deadline = time.monotonic() + 5
        while http.get("/chat/t-4/stream").status_code != 204:
            assert time.monotonic() < deadline, "the finished run is still in flight"
            time.sleep(0.05)


def test_the_chat_route_resumes_a_run_that_waits_for_a_human(booking_app: Served, iq_servers: TestServers) -> None:
    with httpx.Client(base_url=booking_app.url, trust_env=False, timeout=10) as http:
        ticket = http.post("/chat", json={"thread_id": "t-3", "message": "Book a trip to Turin"}).json()
        paused = subscribe(ticket["url"], timeout=20)
        assert paused.terminal is not None and paused.terminal.event == "iq:end"
        last = json.loads(paused.data[-1])
        assert last["type"] == "updates" and last["data"]["__interrupt__"][0]["value"] == {"question": "book Turin?"}

        ticket = http.post("/chat", json={"thread_id": "t-3", "resume": "yes"}).json()
        resumed = subscribe(ticket["url"], timeout=20)
        assert resumed.terminal is not None and resumed.terminal.event == "iq:end"
        parts = [json.loads(data) for data in resumed.data]
        assert [part["data"]["event"] for part in parts if part["type"] == "tools"] == ["tool-started", "tool-finished"]
        history = http.get("/chat/t-3").json()
    assert history[-2:] == [{"type": "tool", "content": "booked Turin: yes"}, {"type": "ai", "content": "Booked."}]


def agui_input(thread_id: str) -> dict[str, Any]:
    return {
        "threadId": thread_id,
        "runId": f"run-{thread_id}",
        "state": {},
        "messages": [{"id": "u1", "role": "user", "content": "What is the weather in Turin?"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def test_the_ag_ui_route_publishes_ag_ui_events(agent_app: Served, iq_servers: TestServers) -> None:
    with httpx.Client(base_url=agent_app.url, trust_env=False, timeout=10) as http:
        ticket = http.post("/agent", json=agui_input("thread-1")).json()
    received = subscribe(ticket["url"], timeout=20)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    types = [json.loads(data)["type"] for data in received.data]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert "TOOL_CALL_RESULT" in types
    [messages] = agent_app.namespace["saved"]
    assert messages[-1].content == FINAL, "before_close saved the answer"


def test_the_ag_ui_route_serves_two_requests_at_once(agent_app: Served, iq_servers: TestServers) -> None:
    """One module-level agent for every request: the relay runs a clone of it."""
    with httpx.Client(base_url=agent_app.url, trust_env=False, timeout=10) as http:
        tickets = [http.post("/agent", json=agui_input(thread_id)).json() for thread_id in ("thread-a", "thread-b")]
    for ticket, thread_id in zip(tickets, ("thread-a", "thread-b"), strict=True):
        received = subscribe(ticket["url"], timeout=20)
        assert received.terminal is not None and received.terminal.event == "iq:end"
        events = [json.loads(data) for data in received.data]
        assert events[0] == {"type": "RUN_STARTED", "threadId": thread_id, "runId": f"run-{thread_id}"}
        assert events[-1]["type"] == "RUN_FINISHED"
        assert "RUN_ERROR" not in [event["type"] for event in events]
