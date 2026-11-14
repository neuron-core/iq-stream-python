"""The browser can read what the adapter publishes: what a real subscriber received is replayed into the real
front-end libraries (tests/frontend/check.mjs), which must rebuild the text, the tool call with its arguments and
result, and how the run ended, for each wire format: AG-UI events, AI SDK UI message chunks and the native chunks.

Needs Node and the JavaScript checkout of @neuron-core/iq-stream, built, with its node_modules: set IQ_STREAM_JS_DIR.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from haystack.components.agents import Agent
from iq_stream import IQClient
from iq_stream.testing import Subscription, TestServers, subscribe_in_background

from haystack_integrations.iq_stream import IQStream
from tests.fakes import ScriptedChatGenerator, text_turn, tool_turn, weather_call
from tests.live import QUESTION, async_weather_tool, failing_agent, recording, weather_agent, weather_tool

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
JS_DIR = os.environ.get("IQ_STREAM_JS_DIR", "")
NEEDED = (
    "@neuron-core/iq-stream/dist/ag-ui",
    "@neuron-core/iq-stream/dist/ai-sdk",
    "@ag-ui/client",
    "@ag-ui/core",
    "ai",
)


def _missing() -> str | None:
    if not JS_DIR:
        return "IQ_STREAM_JS_DIR is not set: point it at a built checkout of neuron-core/iq-stream-js"
    for package in NEEDED:
        if not (Path(JS_DIR) / "node_modules" / package).exists():
            return f"{package} is missing from {JS_DIR}/node_modules: run npm install and the build there"
    if shutil.which("node") is None:
        return "node is not on the PATH"
    return None


pytestmark = pytest.mark.skipif(_missing() is not None, reason=_missing() or "")


@pytest.fixture(scope="module")
def node_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The check in a directory whose node_modules is the JavaScript checkout's."""
    directory = tmp_path_factory.mktemp("frontend")
    shutil.copy(FRONTEND / "check.mjs", directory / "check.mjs")
    (directory / "node_modules").symlink_to(Path(JS_DIR) / "node_modules", target_is_directory=True)
    return directory


def rebuild(node_dir: Path, protocol: str, received: Subscription) -> dict[str, Any]:
    """What the front-end library made of a subscription."""
    path = node_dir / f"recording-{os.getpid()}-{id(received)}.json"
    path.write_text(json.dumps(recording(protocol, received)), encoding="utf-8")
    try:
        process = subprocess.run(
            ["node", "check.mjs", str(path)], cwd=node_dir, capture_output=True, text=True, timeout=60, check=False
        )
    finally:
        path.unlink()
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def published(client: IQClient, servers: TestServers, protocol: str, run: Callable[[IQStream], object]) -> Subscription:
    ticket = client.ticket()
    subscription = subscribe_in_background(servers.default.url, ticket.stream_id, ticket.token)
    try:
        with IQStream(client, ticket.stream_id, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
            run(stream)
    except RuntimeError:
        pass
    return subscription.result(30)


def two_calls_agent() -> Agent:
    calls = [weather_call("Paris", "call_p"), weather_call("Rome", "call_r")]
    script = [
        tool_turn(calls, text=["Let ", "me ", "check."], reasoning=["The user ", "wants weather."]),
        text_turn(["Both ", "are ", "sunny."]),
    ]
    return Agent(chat_generator=ScriptedChatGenerator(script), tools=[weather_tool()])


# -- AG-UI ---------------------------------------------------------------------------------------------------------


def test_ag_ui_stream_agent_rebuilds_the_conversation(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    received = published(
        iq_client, iq_servers, "ag-ui", lambda s: two_calls_agent().run(QUESTION, streaming_callback=s.callback)
    )
    rebuilt = rebuild(node_dir, "ag-ui", received)
    assert rebuilt["invalid"] == [] and rebuilt["failures"] == []
    assert rebuilt["roles"] == ["user", "reasoning", "assistant", "tool", "tool", "assistant"]
    assert rebuilt["texts"] == ["Let me check.", "Both are sunny."]
    assert rebuilt["reasoning"] == ["The user wants weather."]
    assert rebuilt["toolCalls"] == [
        {"id": "call_p", "name": "get_weather", "args": {"city": "Paris"}},
        {"id": "call_r", "name": "get_weather", "args": {"city": "Rome"}},
    ]
    assert rebuilt["toolResults"] == [
        {"toolCallId": "call_p", "content": "Sunny, 21C in Paris"},
        {"toolCallId": "call_r", "content": "Sunny, 21C in Rome"},
    ]
    assert rebuilt["finished"] == {"outcome": None}


def test_ag_ui_a_failed_run_is_one_run_error(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = published(
        iq_client, iq_servers, "ag-ui", lambda s: failing_agent().run(QUESTION, streaming_callback=s.callback)
    )
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    rebuilt = rebuild(node_dir, "ag-ui", received)
    assert rebuilt["runError"] == {"message": "An error occurred.", "code": "error"}
    assert rebuilt["texts"] == ["Hel"]
    # StreamAgent takes the RUN_ERROR as the run's end and does not raise a second error for iq:fail.
    assert rebuilt["failures"] == [] and rebuilt["finished"] is None


def test_ag_ui_a_cancelled_run_finishes_cancelled(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = cancelled_run(iq_client, iq_servers, "ag-ui")
    rebuilt = rebuild(node_dir, "ag-ui", received)
    assert rebuilt["finished"] == {"outcome": {"type": "cancelled"}}
    assert rebuilt["failures"] == []
    assert rebuilt["toolCalls"] == [{"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}]


# -- AI SDK --------------------------------------------------------------------------------------------------------


def test_ai_sdk_use_chat_rebuilds_the_message(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = published(
        iq_client, iq_servers, "ai-sdk", lambda s: two_calls_agent().run(QUESTION, streaming_callback=s.callback)
    )
    rebuilt = rebuild(node_dir, "ai-sdk", received)
    assert rebuilt["invalid"] == []
    assert rebuilt["status"] == "ready" and rebuilt["error"] is None
    assert rebuilt["finish"] == {"isAbort": False, "isError": False}
    assert rebuilt["texts"] == ["Let me check.", "Both are sunny."]
    assert rebuilt["reasoning"] == ["The user wants weather."]
    assert rebuilt["steps"] == 2
    assert [(t["type"], t["toolCallId"], t["state"], t["input"], t["output"]) for t in rebuilt["tools"]] == [
        ("tool-get_weather", "call_p", "output-available", {"city": "Paris"}, "Sunny, 21C in Paris"),
        ("tool-get_weather", "call_r", "output-available", {"city": "Rome"}, "Sunny, 21C in Rome"),
    ]


def test_ai_sdk_a_failed_run_is_the_chat_s_error(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = published(
        iq_client, iq_servers, "ai-sdk", lambda s: failing_agent().run(QUESTION, streaming_callback=s.callback)
    )
    rebuilt = rebuild(node_dir, "ai-sdk", received)
    assert rebuilt["status"] == "error" and rebuilt["error"] == "An error occurred."
    assert rebuilt["texts"] == ["Hel"]


def test_ai_sdk_a_cancelled_run_is_an_abort(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = cancelled_run(iq_client, iq_servers, "ai-sdk")
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    rebuilt = rebuild(node_dir, "ai-sdk", received)
    # chatTransport takes the abort chunk as the run's end, so iq:fail is no second error; useChat keeps the partial
    # message (its isAbort is for a stop pressed in this page).
    assert rebuilt["invalid"] == [] and rebuilt["types"][-1] == "abort"
    assert rebuilt["status"] == "ready" and rebuilt["error"] is None
    assert [(t["toolCallId"], t["state"], t["input"]) for t in rebuilt["tools"]] == [
        ("call_1", "input-available", {"city": "Paris"})
    ]


def test_ai_sdk_a_custom_event_is_a_data_part(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    def run(stream: IQStream) -> None:
        stream.custom("tool_confirmation", {"toolCallId": "call_1", "tool": "get_weather"})
        weather_agent().run(QUESTION, streaming_callback=stream.callback)

    rebuilt = rebuild(node_dir, "ai-sdk", published(iq_client, iq_servers, "ai-sdk", run))
    assert rebuilt["invalid"] == []
    assert rebuilt["data"] == [
        {"type": "data-tool_confirmation", "data": {"toolCallId": "call_1", "tool": "get_weather"}}
    ]


# -- native --------------------------------------------------------------------------------------------------------


def test_native_chunks_read_by_the_transport_client(
    node_dir: Path, iq_client: IQClient, iq_servers: TestServers
) -> None:
    received = published(
        iq_client, iq_servers, "haystack", lambda s: weather_agent().run(QUESTION, streaming_callback=s.callback)
    )
    rebuilt = rebuild(node_dir, "haystack", received)
    assert rebuilt["outcome"] == {"kind": "end"}
    assert rebuilt["text"] == "Let me check.It is sunny in Paris."
    assert rebuilt["toolCalls"] == [{"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}]
    assert rebuilt["toolResults"] == [{"id": "call_1", "result": "Sunny, 21C in Paris", "error": False}]
    assert rebuilt["types"][0] == "start" and rebuilt["last"] == {"type": "finish"}


def test_native_a_failed_run_ends_with_its_error(node_dir: Path, iq_client: IQClient, iq_servers: TestServers) -> None:
    received = published(
        iq_client, iq_servers, "haystack", lambda s: failing_agent().run(QUESTION, streaming_callback=s.callback)
    )
    rebuilt = rebuild(node_dir, "haystack", received)
    assert rebuilt["last"] == {"type": "error", "message": "An error occurred."}
    assert rebuilt["outcome"] == {"kind": "fail", "reason": "publisher"}


def cancelled_run(client: IQClient, servers: TestServers, protocol: str) -> Subscription:
    async def main() -> Subscription:
        started = asyncio.Event()
        script = [tool_turn([weather_call()]), text_turn(["never"])]
        agent = Agent(chat_generator=ScriptedChatGenerator(script), tools=[async_weather_tool(30, started)])
        ticket = client.ticket()
        subscription = subscribe_in_background(servers.default.url, ticket.stream_id, ticket.token)

        async def run() -> None:
            async with IQStream(client, ticket.stream_id, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
                await agent.run_async(QUESTION, streaming_callback=stream.acallback)

        task = asyncio.create_task(run())
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await subscription

    return asyncio.run(main())
