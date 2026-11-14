"""Every Python block of the README is run here, against the default test server.

Scripts run in a process of their own after a prelude that defines what the README says is the application's
(``run_agent()``, ``save_answer()``, ...), with the README's placeholder addresses replaced by the test server's and
the credentials in the environment. They must exit cleanly and print nothing to stderr, where a publisher that gave
up would have logged its warning. Blocks that hold tests run under pytest, with the README's ``conftest.py``.
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
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)
CONFTESTS = [block for block in BLOCKS if block.startswith("# conftest.py")]
TESTS = [block for block in BLOCKS if "def test_" in block]
SCRIPTS = [block for block in BLOCKS if block not in CONFTESTS and block not in TESTS]
PLACEHOLDERS = ("http://iq.internal:8080", "https://stream.example.com")

PRELUDE = """
def run_agent(messages):
    for delta in ["It ", "is ", "sunny."]:
        yield {"type": "text-delta", "delta": delta}

async def arun_agent(messages):
    for chunk in run_agent(messages):
        yield chunk

def save_answer(messages):
    pass

async def asave_answer(messages):
    pass

messages = [{"role": "user", "content": "What is the weather?"}]
"""

EPILOGUE = """
from iq_stream import AsyncPublisher as _AsyncPublisher, Publisher as _Publisher
for _value in list(globals().values()):
    if isinstance(_value, (_Publisher, _AsyncPublisher)):
        assert not _value.alive and _value.error is None, _value
"""


def _environment(servers: TestServers) -> dict[str, str]:
    return {
        **os.environ,
        "IQ_STREAM_URL": servers.default.url,
        "IQ_STREAM_PUBLIC_URL": servers.default.url,
        "IQ_STREAM_APP_KEY": servers.app_key,
        "IQ_STREAM_APP_SECRET": servers.app_secret,
    }


def test_the_readme_has_samples_of_each_kind() -> None:
    assert len(SCRIPTS) >= 4
    assert len(CONFTESTS) == 1
    assert len(TESTS) == 2


@pytest.mark.parametrize("block", SCRIPTS, ids=[f"script-{n}" for n in range(len(SCRIPTS))])
def test_a_readme_script_runs(block: str, iq_servers: TestServers) -> None:
    code = block
    for placeholder in PLACEHOLDERS:
        code = code.replace(placeholder, iq_servers.default.url)
    process = subprocess.run(
        [sys.executable, "-c", PRELUDE + code + EPILOGUE],
        env=_environment(iq_servers),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert process.stderr == ""


def test_the_readme_tests_pass(iq_servers: TestServers, tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(CONFTESTS[0], encoding="utf-8")
    for n, block in enumerate(TESTS):
        (tmp_path / f"test_readme_{n}.py").write_text(block, encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:asyncio",
            "-W",
            "error",
        ],
        cwd=tmp_path,
        env=_environment(iq_servers),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "3 passed" in process.stdout
