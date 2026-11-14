"""The front-end contract: what a real subscriber received, fed to the browser side of @neuron-core/iq-stream in Node
(tests/frontend/check.mjs), reconstructs the answer, the tool call with its arguments and result, and the outcome."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from agno.agent import Agent
from iq_stream import IQClient
from iq_stream.testing import TestServers, subscribe_in_background
from live import recording
from scripted_model import ANSWER, SLOW, weather_agent

from iq_stream_agno import MASKED_ERROR, apublish, start_run

CHECK = Path(__file__).resolve().parents[1] / "frontend" / "check.mjs"
JS_DIR = os.environ.get("IQ_STREAM_JS_DIR", "")
BUILT = Path(JS_DIR, "node_modules", "@neuron-core", "iq-stream", "dist", "ag-ui", "index.js")

pytestmark = pytest.mark.skipif(
    not JS_DIR or not BUILT.is_file() or shutil.which("node") is None,
    reason="set IQ_STREAM_JS_DIR to a built checkout of iq-stream-js (with node_modules) and put node on PATH",
)


def check(record: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    done = subprocess.run(
        ["node", str(CHECK), str(path)],
        env={**os.environ, "IQ_STREAM_JS_DIR": JS_DIR},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


async def published(
    client: IQClient, servers: TestServers, agent: Agent, protocol: str, cancel: bool = False
) -> dict[str, Any]:
    ticket = client.ticket()
    subscription = subscribe_in_background(servers.default.url, ticket.stream_id, ticket.token)
    stream = agent.arun("Weather in Paris?", stream=True, stream_events=True, run_id=ticket.stream_id)
    async for event in apublish(stream, client.apublisher(ticket.stream_id), protocol=protocol):  # type: ignore[arg-type]
        if cancel and event.event == "ToolCallStarted":
            Agent.cancel_run(ticket.stream_id)
    return recording(await subscription, protocol)


def result_text(content: str) -> str:
    try:
        decoded = json.loads(content)
    except ValueError:
        return content
    return decoded if isinstance(decoded, str) else content


async def test_native_frames_rebuild_the_answer_and_the_tool_call(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    seen = check(await published(iq_client, iq_servers, weather_agent(), "agno"), tmp_path)
    assert seen["text"] == "Let me check the weather. " + ANSWER
    assert seen["tools"] == [{"name": "get_weather", "args": {"city": "Paris"}, "result": "21C and sunny in Paris"}]
    assert seen["last"] == "RunCompleted" and seen["outcome"] == {"type": "end"}


async def test_native_frames_of_a_failed_run_end_with_run_error_and_fail(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    seen = check(await published(iq_client, iq_servers, weather_agent(fail_on_turn=2), "agno"), tmp_path)
    assert seen["last"] == "RunError" and seen["lastContent"] == MASKED_ERROR
    assert seen["outcome"] == {"type": "fail", "reason": "publisher"}


async def test_ag_ui_frames_rebuild_the_messages_in_stream_agent(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    seen = check(await published(iq_client, iq_servers, weather_agent(), "agui"), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"] == "Let me check the weather. " + ANSWER
    (call,) = seen["toolCalls"]
    assert call["name"] == "get_weather" and call["args"] == {"city": "Paris"}
    (result,) = seen["toolResults"]
    assert result["toolCallId"] == call["id"] and result_text(result["content"]) == "21C and sunny in Paris"
    assert seen["last"]["type"] == "RUN_FINISHED"


async def test_ag_ui_frames_of_a_failed_run_end_in_run_error_without_a_second_error(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    seen = check(await published(iq_client, iq_servers, weather_agent(fail_on_turn=2), "agui"), tmp_path)
    assert seen["last"]["type"] == "RUN_ERROR" and seen["last"]["message"] == MASKED_ERROR
    assert not any("stream failed" in failure for failure in seen["failures"])
    assert seen["text"].startswith("Let me check the weather.")


async def test_ag_ui_frames_of_a_cancelled_run_finish_with_the_cancelled_outcome(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    SLOW["seconds"] = 0.3
    agent = weather_agent(tool_name="slow_weather")
    seen = check(await published(iq_client, iq_servers, agent, "agui", cancel=True), tmp_path)
    assert seen["rejected"] is None and not any("stream failed" in failure for failure in seen["failures"])
    assert seen["last"]["type"] == "RUN_FINISHED" and seen["last"]["outcome"] == {"type": "cancelled"}


async def test_a_start_run_stream_reaches_stream_agent(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handle = start_run(iq_client, weather_agent(), "Weather in Paris?", protocol="agui")
    subscription = subscribe_in_background(handle.url)
    await handle.task
    seen = check(recording(await asyncio.to_thread(subscription.result, 15), "agui"), tmp_path)
    assert seen["text"] == "Let me check the weather. " + ANSWER and seen["last"]["type"] == "RUN_FINISHED"
