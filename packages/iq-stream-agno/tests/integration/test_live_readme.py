"""Every Python block of the README runs here, against the default test server, in a process of its own.

A prelude defines what the README says is the application's: ``agent``, an Agno agent on the scripted model. A block
that defines ``app`` (the FastAPI route, AgentOS) is served with uvicorn and driven over HTTP by an epilogue, which
subscribes to the ticket the route answered with and checks that the run arrived and ended.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from iq_stream.testing import TestServers

README = Path(__file__).resolve().parents[2] / "README.md"
TESTS = Path(__file__).resolve().parents[1]
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)

PRELUDE = f"""
import sys
sys.path.insert(0, {str(TESTS)!r})
from scripted_model import weather_agent
agent = weather_agent()
"""

SERVE = """
import threading, time
import httpx, uvicorn
from iq_stream.testing import free_port, subscribe

_port = free_port()
_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=_port, log_level="warning"))
_thread = threading.Thread(target=_server.run, daemon=True)
_thread.start()
while not _server.started:
    time.sleep(0.02)
try:
    _path = "/iq/runs" if "AgentOS" in repr(type(agent_os)) else "/chat"
    _ticket = httpx.post(f"http://127.0.0.1:{_port}{_path}", json={"message": "Weather?"}, trust_env=False).json()
    _received = subscribe(_ticket["url"], timeout=20)
    assert _received.terminal is not None and _received.terminal.event == "iq:end", _received.terminal
    assert _received.frames and '"RunCompleted"' in _received.frames[-1].data
finally:
    _server.should_exit = True
    _thread.join(20)
"""


def test_the_readme_has_its_samples() -> None:
    assert len(BLOCKS) == 4
    assert sum("app = " in block for block in BLOCKS) == 2


@pytest.mark.parametrize("block", BLOCKS, ids=[f"block-{n}" for n in range(len(BLOCKS))])
def test_a_readme_block_runs(block: str, iq_servers: TestServers) -> None:
    epilogue = ""
    if "app = " in block:
        epilogue = SERVE if "AgentOS(" in block else "agent_os = None\n" + SERVE
    process = subprocess.run(
        [sys.executable, "-c", PRELUDE + block + epilogue],
        env={
            **os.environ,
            "AGNO_TELEMETRY": "false",
            "IQ_STREAM_URL": iq_servers.default.url,
            "IQ_STREAM_PUBLIC_URL": iq_servers.default.url,
            "IQ_STREAM_APP_KEY": iq_servers.app_key,
            "IQ_STREAM_APP_SECRET": iq_servers.app_secret,
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "Traceback" not in process.stderr and "iq_stream" not in process.stderr, process.stderr
