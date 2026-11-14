"""Every Python block of the README runs here, against the default test server, in a process of its own.

A prelude defines what the README says is the application's: ``agent`` (the fake-model weather agent),
``approval_agent`` (one whose tool waits for a human), ``save_answer()`` and ``answer_from_the_browser()``. A block that
defines ``app`` is served with uvicorn and driven over HTTP by an epilogue, which subscribes with the ticket the route
answered and checks that the run arrived and ended. A block that holds a test runs under pytest.
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
from fakes import hitl_agent, weather_agent

agent = weather_agent()
approval_agent = hitl_agent()


async def save_answer(outcome):
    print("saved", outcome.status)


async def answer_from_the_browser(event):
    return "yes"
"""

SERVE = """
import json as _json
import threading as _threading
import time as _time

import httpx as _httpx
import uvicorn as _uvicorn
from iq_stream.testing import free_port as _free_port, subscribe as _subscribe

_port = _free_port()
_server = _uvicorn.Server(_uvicorn.Config(app, host="127.0.0.1", port=_port, log_level="warning", lifespan="off"))
_thread = _threading.Thread(target=_server.run, daemon=True)
_thread.start()
while not _server.started:
    _time.sleep(0.02)
try:
    _body = {"thread_id": "thread-1", "message": "Weather?"}
    _ticket = _httpx.post(f"http://127.0.0.1:{_port}/chat", json=_body, trust_env=False, timeout=10).json()
    _received = _subscribe(_ticket["url"], timeout=20)
    assert _received.terminal is not None and _received.terminal.event == "iq:end", _received.terminal
    assert _json.loads(_received.frames[-1].data)["type"] == "RUN_FINISHED"
    _deadline = _time.monotonic() + 5
    while running and _time.monotonic() < _deadline:
        _time.sleep(0.02)
    assert not running
finally:
    _server.should_exit = True
    _thread.join(20)
"""


def environment(servers: TestServers) -> dict[str, str]:
    return {
        **os.environ,
        "IQ_STREAM_URL": servers.default.url,
        "IQ_STREAM_APP_KEY": servers.app_key,
        "IQ_STREAM_APP_SECRET": servers.app_secret,
    }


def test_the_readme_has_its_samples() -> None:
    assert len(BLOCKS) == 4
    assert sum("app = FastAPI()" in block for block in BLOCKS) == 1
    assert sum("def test_" in block for block in BLOCKS) == 1


@pytest.mark.parametrize("block", BLOCKS, ids=[f"block-{n}" for n in range(len(BLOCKS))])
def test_a_readme_block_runs(block: str, iq_servers: TestServers, tmp_path: Path) -> None:
    script = tmp_path / ("test_sample.py" if "def test_" in block else "sample.py")
    script.write_text(PRELUDE + block + (SERVE if "app = FastAPI()" in block else ""), encoding="utf-8")
    command = [sys.executable, str(script)]
    if "def test_" in block:
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(script),
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "asyncio_mode=auto",
        ]
    done = subprocess.run(
        command, cwd=tmp_path, env=environment(iq_servers), capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
    if "def test_" in block:
        assert "1 passed" in done.stdout
    else:
        assert done.stderr == ""  # a publisher that gave up would have logged its warning there
    if "async def main()" in block:
        assert "saving completed It is 21 °C and sunny in Paris." in done.stdout
        assert "completed" in done.stdout.splitlines()[-1]
    if "run_with_approval" in block:
        assert done.stdout.strip().splitlines()[-1] == "completed"
