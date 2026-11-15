"""The browser's side of the contract: what a real subscriber received from the server, fed to the real front-end
library of ``@neuron-core/iq-stream`` in Node (tests/frontend/replay.mjs), which says what it reconstructed.

Needs ``IQ_STREAM_JS_DIR``, the iq-stream-js checkout with its packages built and installed, and Node.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from crew_fakes import ANSWER, TERM, FakeMemoryLLM, LookupTool, embed, make_crew
from crewai.memory.unified_memory import Memory
from iq_stream import IQClient
from iq_stream.testing import Subscription, TestServers, subscribe_in_background

from iq_stream_crewai import EVENT_TYPES, publish

SCRIPT = Path(__file__).resolve().parents[1] / "frontend" / "replay.mjs"
JS_DIR = os.environ.get("IQ_STREAM_JS_DIR", "")
NODE = shutil.which("node")
BUILT = Path(JS_DIR, "packages", "iq-stream", "dist", "ag-ui", "index.js")

pytestmark = pytest.mark.skipif(
    not JS_DIR or NODE is None or not BUILT.is_file(),
    reason="the front-end contract needs IQ_STREAM_JS_DIR (the iq-stream-js checkout, built and installed) and node",
)


def replay(protocol: str, received: Subscription, tmp_path: Path) -> dict[str, Any]:
    assert received.terminal is not None
    capture = {
        "protocol": protocol,
        "frames": [{"id": frame.id, "event": frame.event, "data": frame.data} for frame in received.frames],
        "terminal": {"event": received.terminal.event, "data": received.terminal.data},
        "events": sorted(EVENT_TYPES),
    }
    path = tmp_path / f"{protocol}.json"
    path.write_text(json.dumps(capture), encoding="utf-8")
    assert NODE is not None
    process = subprocess.run(
        [NODE, str(SCRIPT), str(path)],
        env={**os.environ, "IQ_STREAM_JS_DIR": JS_DIR},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout.strip().splitlines()[-1])


def run(iq: IQClient, servers: TestServers, protocol: str, **crew: Any) -> Subscription:
    ticket = iq.ticket()
    subscription = subscribe_in_background(servers.default.url, ticket.stream_id, ticket.token)
    try:
        with publish(iq, ticket.stream_id, protocol=protocol):  # type: ignore[arg-type]
            make_crew(**crew).kickoff()
    except RuntimeError:
        pass
    return subscription.result(30)


def test_ag_ui_a_run_with_a_tool_call(iq_client: IQClient, iq_servers: TestServers, tmp_path: Path) -> None:
    result = replay("ag-ui", run(iq_client, iq_servers, "ag-ui"), tmp_path)
    assert result["error"] is None and result["outcome"] == "success"
    assert result["types"][0] == "RUN_STARTED" and result["types"][-1] == "RUN_FINISHED"
    [call, tool, text] = result["messages"]
    assert call["role"] == "assistant"
    [tool_call] = call["toolCalls"]
    assert tool_call["name"] == "lookup" and json.loads(tool_call["arguments"]) == {"term": TERM}
    assert tool["role"] == "tool" and tool["content"] == f"value:{TERM}"
    assert text == {"role": "assistant", "content": ANSWER}


def test_ag_ui_a_crew_with_memory(iq_client: IQClient, iq_servers: TestServers, tmp_path: Path) -> None:
    # Memory's analysis answers JSON through model calls of its own: none of it is a message of the conversation.
    memory = Memory(llm=FakeMemoryLLM(), embedder=embed, storage=str(tmp_path / "memory"))
    crew = make_crew()
    crew.memory = memory
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    try:
        with publish(iq_client, ticket.stream_id):
            crew.kickoff()
    finally:
        memory.close()
    result = replay("ag-ui", subscription.result(30), tmp_path)
    assert result["error"] is None and result["outcome"] == "success"
    assert [message.get("content") for message in result["messages"] if message["role"] == "assistant"] == [
        None,
        ANSWER,
    ]
    assert result["messages"][-1] == {"role": "assistant", "content": ANSWER}


def test_ag_ui_a_failing_run(iq_client: IQClient, iq_servers: TestServers, tmp_path: Path) -> None:
    received = run(iq_client, iq_servers, "ag-ui", fail_on_call=2)
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    result = replay("ag-ui", received, tmp_path)
    # RUN_ERROR, then iq:fail with reason publisher: the run ends once, on the first.
    assert result["error"] is None
    assert result["outcome"] == "error"
    assert result["runError"] == {"message": "An error occurred.", "code": "RuntimeError"}
    assert result["messages"][-1]["content"] == "Let me "


async def test_ag_ui_a_cancelled_run(iq_client: IQClient, iq_servers: TestServers, tmp_path: Path) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    gate = threading.Event()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    class Signalling(LookupTool):
        def _run(self, term: str = "default") -> str:
            loop.call_soon_threadsafe(started.set)
            return super()._run(term)

    async def go() -> None:
        async with publish(iq_client, ticket.stream_id):
            await make_crew(tool=Signalling(gate=gate, seconds=10.0)).akickoff()

    task = asyncio.create_task(go())
    await asyncio.wait_for(started.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    received = await subscription.aresult(30)
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    result = replay("ag-ui", received, tmp_path)
    assert result["error"] is None and result["outcome"] == "cancelled"
    [call] = result["messages"]
    assert call["toolCalls"][0]["name"] == "lookup"


def test_crewai_native_frames(iq_client: IQClient, iq_servers: TestServers, tmp_path: Path) -> None:
    received = run(iq_client, iq_servers, "crewai")
    result = replay("crewai", received, tmp_path)
    assert result["outcome"] == {"type": "end", "lastEventId": len(received.frames)}
    assert result["ids"] == [frame.id for frame in received.frames]
    assert result["names"] == [frame.event for frame in received.frames]
    assert result["texts"] == [ANSWER]
    assert result["tools"] == [{"name": "lookup", "args": {"term": TERM}, "output": f"value:{TERM}"}]


def test_crewai_native_frames_of_a_failing_run(iq_client: IQClient, iq_servers: TestServers, tmp_path: Path) -> None:
    received = run(iq_client, iq_servers, "crewai", fail_on_call=2, tool=LookupTool(explode=True))
    result = replay("crewai", received, tmp_path)
    assert result["outcome"]["type"] == "fail" and result["outcome"]["reason"] == "publisher"
    assert result["tools"] == [{"name": "lookup", "args": {"term": TERM}, "error": "An error occurred."}]
    assert result["names"][-1] == "crew_kickoff_failed"
