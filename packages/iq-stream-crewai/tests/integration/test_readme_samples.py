"""Every Python block of the README but the route (run over HTTP in test_route_http.py), against the default
server, each in a process of its own after a prelude that defines the README's ``crew``, ``build_crew()`` and
``save_answer()`` on a fake model. A block that holds a test is run by calling it."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest
from iq_stream.testing import TestServers
from readme_blocks import PRELUDE, SCRIPTS

EPILOGUE = """
from crew_fakes import ANSWER as _ANSWER
from iq_stream_crewai import CrewAIStream as _CrewAIStream

assert all(text == _ANSWER for text in saved.values()), saved
for _value in list(globals().values()):
    if isinstance(_value, _CrewAIStream):
        assert _value.result is not None and _value.result.outcome == "end", _value.result
        assert _value.errors == [] and _value.publisher.error is None
print("saved", len(saved))
"""


def test_the_readme_has_its_samples() -> None:
    assert len(SCRIPTS) == 4
    assert sum("def test_" in block for block in SCRIPTS) == 1


@pytest.mark.parametrize("block", SCRIPTS, ids=[f"block-{number}" for number in range(len(SCRIPTS))])
def test_a_readme_block_runs(block: str, iq_servers: TestServers, tmp_path: os.PathLike[str]) -> None:
    code = PRELUDE + block
    for test in re.findall(r"^def (test_\w+)\(", block, re.MULTILINE):
        code += f"\n{test}()\n"
    environment = {
        **os.environ,
        "IQ_STREAM_URL": iq_servers.default.url,
        "IQ_STREAM_APP_KEY": iq_servers.app_key,
        "IQ_STREAM_APP_SECRET": iq_servers.app_secret,
        "CREWAI_DISABLE_TELEMETRY": "true",
        "OTEL_SDK_DISABLED": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "XDG_DATA_HOME": str(tmp_path),
    }
    process = subprocess.run(
        [sys.executable, "-c", code + EPILOGUE], env=environment, capture_output=True, text=True, timeout=120
    )
    assert process.returncode == 0, process.stderr[-3000:]
    # A publisher that gave up, or an adapter failure, would have logged a warning.
    assert "iq-stream" not in process.stderr and "iq_stream" not in process.stderr, process.stderr[-3000:]
    saving = "save_answer(" in block
    assert process.stdout.rstrip().endswith(f"saved {1 if saving else 0}")
