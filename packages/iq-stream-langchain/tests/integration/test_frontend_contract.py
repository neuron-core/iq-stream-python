"""The browser can read what the adapter publishes: what a real subscriber received is fed to the real front-end
libraries (tests/frontend/contract.mjs), which must rebuild the text, the tool call with its arguments and result,
and how the run ended.

Needs Node and the JavaScript checkout of @neuron-core/iq-stream with its node_modules: set IQ_STREAM_JS_DIR to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FINAL,
    FIRST,
    TOOL_RESULT,
    USER,
    FakeToolCallingModel,
    booking_agent,
    failing_agent,
    research_supervisor,
    tool_call,
    weather_agent,
)
from iq_stream import IQClient
from iq_stream.testing import TestServers, subscribe
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from streams import publish_agui, recording, watch

from iq_stream_langchain import MASKED_MESSAGE, apublish_run, publish_run

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
README = Path(__file__).resolve().parents[2] / "README.md"
JS_DIR = os.environ.get("IQ_STREAM_JS_DIR")


def _missing() -> str | None:
    if not JS_DIR:
        return "IQ_STREAM_JS_DIR is not set: point it at the checkout of neuron-core/iq-stream-js"
    modules = Path(JS_DIR) / "node_modules"
    for package in ("@neuron-core/iq-stream/dist/ag-ui", "@ai-sdk/langchain", "@langchain/langgraph-sdk", "ai"):
        if not (modules / package).exists():
            return f"{modules / package} is missing: run npm install and the build in {JS_DIR}"
    if shutil.which("node") is None:
        return "node is not on the PATH"
    return None


pytestmark = pytest.mark.skipif(_missing() is not None, reason=_missing() or "")


@pytest.fixture(scope="module")
def node_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scripts in a directory whose node_modules is the JavaScript checkout's."""
    directory = tmp_path_factory.mktemp("frontend")
    for script in FRONTEND.glob("*.mjs"):
        shutil.copy(script, directory / script.name)
    (directory / "node_modules").symlink_to(Path(str(JS_DIR)) / "node_modules", target_is_directory=True)
    return directory


def rebuild(node_dir: Path, check: str, received: dict[str, Any], *arguments: str) -> dict[str, Any]:
    """What the front-end library made of the recording."""
    path = node_dir / f"recording-{os.getpid()}-{id(received)}.json"
    path.write_text(json.dumps(received), encoding="utf-8")
    try:
        process = subprocess.run(
            ["node", "contract.mjs", check, str(path), *arguments],
            cwd=node_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        path.unlink()
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.strip().splitlines()[-1])


def published(client: IQClient, servers: TestServers, agent: Any, config: Any = None) -> dict[str, Any]:
    ticket, subscription = watch(client, servers.default.url)
    try:
        with client.publisher(ticket.stream_id) as publisher:
            publish_run(publisher, agent, USER, config)
    except RuntimeError:
        pass
    return recording(subscription.result(10))


async def cancelled(client: IQClient, servers: TestServers) -> dict[str, Any]:
    ticket, subscription = watch(client, servers.default.url)

    async def run() -> None:
        async with client.apublisher(ticket.stream_id) as publisher:
            await apublish_run(publisher, weather_agent(token_delay=0.05), USER)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return recording(await subscription)


EXPECTED_THREAD = [
    {
        "type": "ai",
        "content": FIRST,
        "tool_calls": [{"name": "get_weather", "args": {"city": "Turin"}, "id": "call_1"}],
    },
    {"type": "tool", "content": TOOL_RESULT, "tool_calls": []},
    {"type": "ai", "content": FINAL, "tool_calls": []},
]


def test_use_stream_rebuilds_the_thread(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    result = rebuild(node_dir, "usestream", published(iq_client, iq_servers, weather_agent()))
    assert result["error"] is None and result["finished"] is True
    assert result["messages"] == EXPECTED_THREAD


def test_use_stream_reports_a_failed_run_by_its_error_name(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    result = rebuild(node_dir, "usestream", published(iq_client, iq_servers, failing_agent()))
    assert result["error"] == {"name": "RuntimeError", "message": MASKED_MESSAGE}
    assert result["messages"][0]["content"] == "This answer will "


async def test_use_stream_tells_a_cancel_from_a_failure(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    result = rebuild(node_dir, "usestream", await cancelled(iq_client, iq_servers))
    assert result["error"] == {"name": "Cancelled", "message": "run cancelled"}


def test_use_stream_sees_the_interrupt(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    @tool
    def book(city: str) -> str:
        """Book a trip; a human approves it first."""
        return f"booked {city}: {interrupt({'question': f'book {city}?'})}"

    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call("book", {"city": "Turin"}, "c1")])])  # type: ignore[list-item]
    )
    agent = create_agent(model, [book], checkpointer=InMemorySaver())
    result = rebuild(
        node_dir, "usestream", published(iq_client, iq_servers, agent, {"configurable": {"thread_id": "t"}})
    )
    assert result["finished"] is True and result["error"] is None
    assert [item["value"] for item in result["interrupt"]] == [{"question": "book Turin?"}]


def test_the_readme_use_stream_transport_starts_and_resumes_a_run(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    agent = booking_agent(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "resume"}}
    assert "__interrupt__" in agent.invoke(USER, config), "the thread waits for a human"
    ticket, subscription = watch(iq_client, iq_servers.default.url)
    with iq_client.publisher(ticket.stream_id) as publisher:
        publish_run(publisher, agent, Command(resume="yes"), config)

    result = rebuild(node_dir, "transport", recording(subscription.result(10)), str(README))
    assert result["posted"] == [
        {"url": "/chat", "body": {"thread_id": "thread-1", "message": "Book a trip to Turin"}},
        {"url": "/chat", "body": {"thread_id": "thread-1", "resume": "yes"}},
    ]
    assert result["error"] is None and result["finished"] is True
    assert result["messages"] == [
        {"type": "tool", "content": "booked Turin: yes", "tool_calls": []},
        {"type": "ai", "content": "Booked.", "tool_calls": []},
    ]


def test_use_chat_rebuilds_the_answer_with_its_tool_call(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    result = rebuild(node_dir, "usechat", published(iq_client, iq_servers, weather_agent()))
    assert result["status"] == "ready" and result["error"] is None
    assert result["role"] == "assistant"
    assert result["parts"] == [
        {"type": "text", "text": FIRST},
        {
            "type": "dynamic-tool",
            "toolName": "get_weather",
            "state": "output-available",
            "input": {"city": "Turin"},
            "output": TOOL_RESULT,
        },
        {"type": "text", "text": FINAL},
    ]
    assert result["chunks"][0] == "start" and result["chunks"][-1] == "finish"


def test_use_chat_shows_a_failed_run_once(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    result = rebuild(node_dir, "usechat", published(iq_client, iq_servers, failing_agent()))
    assert result["status"] == "error" and result["error"] == MASKED_MESSAGE
    assert result["chunks"][-1] == "error"


async def test_use_chat_shows_a_cancelled_run_as_aborted(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    result = rebuild(node_dir, "usechat", await cancelled(iq_client, iq_servers))
    assert result["chunks"][-1] == "abort"
    assert result["status"] == "ready" and result["error"] is None


def test_use_chat_reports_a_stream_that_is_gone(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    ticket = iq_client.ticket()
    with iq_client.publisher(ticket.stream_id) as publisher:
        publish_run(publisher, weather_agent(), USER)
    late = subscribe(iq_servers.default.url, ticket.stream_id, ticket.token, timeout=10)
    result = rebuild(node_dir, "usechat", recording(late))
    assert result["status"] == "error" and "gone" in result["error"]


async def test_stream_agent_rebuilds_the_messages(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = await publish_agui(iq_client, iq_servers.default.url, weather_agent(checkpointer=InMemorySaver()))
    result = rebuild(node_dir, "agui", recording(received))
    assert result["failures"] == [] and result["runError"] is None and result["outcome"] is None
    assert result["messages"][1:] == [
        {
            "role": "assistant",
            "content": FIRST,
            "toolCalls": [{"name": "get_weather", "arguments": '{"city": "Turin"}'}],
            "toolCallId": None,
        },
        {"role": "tool", "content": TOOL_RESULT, "toolCalls": [], "toolCallId": "call_1"},
        {"role": "assistant", "content": FINAL, "toolCalls": [], "toolCallId": None},
    ]
    assert result["events"][-1] == "RUN_FINISHED"


async def test_stream_agent_reports_a_failed_run_once(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    received = await publish_agui(iq_client, iq_servers.default.url, failing_agent(checkpointer=InMemorySaver()))
    result = rebuild(node_dir, "agui", recording(received))
    assert result["runError"] == MASKED_MESSAGE
    assert result["failures"] == [], "the iq:fail after RUN_ERROR is not a second error"


async def test_stream_agent_accepts_a_cancelled_run(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    graph = weather_agent(token_delay=0.05, checkpointer=InMemorySaver())
    received = await publish_agui(iq_client, iq_servers.default.url, graph, cancel_after=0.3)
    result = rebuild(node_dir, "agui", recording(received))
    assert result["outcome"] == {"type": "cancelled"}
    assert result["failures"] == [], "the client's checks accepted the events that closed what was open"
    assert result["events"][-1] == "RUN_FINISHED"


async def test_stream_agent_accepts_a_run_cancelled_inside_a_sub_agent(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    received = await publish_agui(
        iq_client, iq_servers.default.url, research_supervisor(), cancel_after=0.5, subagent_visibility="attributed"
    )
    events = [json.loads(frame.data)["type"] for frame in received.frames]
    assert "SUBAGENT_STARTED" in events and "SUBAGENT_FINISHED" not in events, "cancelled while the sub-agent wrote"
    result = rebuild(node_dir, "agui", recording(received))
    assert result["outcome"] == {"type": "cancelled"}
    assert result["failures"] == [], "the client's checks accepted the events that closed the sub-agent"
