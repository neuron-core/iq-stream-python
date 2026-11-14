"""Every Python block of the README runs here, against the default test server, in a process of its own.

A prelude defines what the README says is the application's: ``agent`` (the scripted weather agent),
``chat_generator``, ``get_weather``, ``load_messages()``, ``save_messages()``, ``wait_for_answer()``, and the client
``iq`` that the first block defines. A script must exit cleanly and print nothing to stderr, where a stream that
stopped publishing logs its warning; the streams it leaves behind must be closed by the server, and it must save what
it says it saves. The Hayhooks block is an excerpt of the example wrapper that test_live_hayhooks.py deploys: its parts
must be in that file as they are.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from iq_stream.testing import TestServers

PACKAGE = Path(__file__).resolve().parents[2]
README = (PACKAGE / "README.md").read_text(encoding="utf-8")
BLOCKS = re.findall(r"```python\n(.*?)```", README, re.DOTALL)
EXCERPTS = [block for block in BLOCKS if "class PipelineWrapper" in block]
RUNNABLE = [block for block in BLOCKS if block not in EXCERPTS]

PRELUDE = """
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from iq_stream import IQClient

from tests.fakes import ScriptedChatGenerator, weather_script
from tests.live import weather_tool

iq = IQClient()  # the README's first block defines it; the later ones use it
get_weather = weather_tool()
chat_generator = ScriptedChatGenerator(weather_script())
agent = Agent(chat_generator=ScriptedChatGenerator(weather_script()), tools=[get_weather])
SAVED = {}


def load_messages(chat_id):
    return [ChatMessage.from_user("Weather in Paris?")]


def save_messages(chat_id, messages):
    SAVED[chat_id] = [message.to_dict() for message in messages]


async def asave_messages(chat_id, messages):
    save_messages(chat_id, messages)


def wait_for_answer(tool_call_id):
    return True
"""

EPILOGUE = """
from haystack_integrations.iq_stream import IQStream as _IQStream

for _value in list(globals().values()):
    if isinstance(_value, _IQStream):
        assert _value.result is not None and _value.result.closed and not _value.failed, _value
"""


def test_the_readme_has_its_samples() -> None:
    assert len(EXCERPTS) == 1
    assert len(RUNNABLE) == 6
    assert sum("def test_" in block for block in RUNNABLE) == 1


def test_the_hayhooks_block_is_an_excerpt_of_the_tested_example() -> None:
    example = (PACKAGE / "examples" / "hayhooks" / "weather_agent" / "pipeline_wrapper.py").read_text(encoding="utf-8")
    for part in re.split(r"^\.\.\.\n", EXCERPTS[0], flags=re.MULTILINE):
        assert part in example, part


@pytest.mark.parametrize("block", RUNNABLE, ids=[f"block{n}" for n in range(len(RUNNABLE))])
def test_a_readme_block_runs(block: str, iq_servers: TestServers, tmp_path: Path) -> None:
    env = {
        **os.environ,
        "IQ_STREAM_URL": iq_servers.default.url,
        "IQ_STREAM_PUBLIC_URL": iq_servers.default.url,
        "IQ_STREAM_APP_KEY": iq_servers.app_key,
        "IQ_STREAM_APP_SECRET": iq_servers.app_secret,
        "HAYSTACK_TELEMETRY_ENABLED": "False",
        "PYTHONPATH": str(PACKAGE),
    }
    if "def test_" in block:
        script = tmp_path / "test_sample.py"
        script.write_text(PRELUDE + block, encoding="utf-8")
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(script)]
    else:
        script = tmp_path / "sample.py"
        saves = '\nassert SAVED, "nothing was saved"\n' if "save_messages(" in block else ""
        script.write_text(PRELUDE + block + EPILOGUE + saves, encoding="utf-8")
        command = [sys.executable, str(script)]
    process = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    if "def test_" not in block:
        assert process.stderr == "", process.stderr
