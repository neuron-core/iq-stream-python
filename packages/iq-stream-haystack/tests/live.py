"""What the integration tests share: agents on the scripted model, and the checks of what a subscriber received."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool, tool
from iq_stream.testing import Subscription

from tests.fakes import ScriptedChatGenerator, Turn, text_turn, weather_script

QUESTION = [ChatMessage.from_user("Weather in Paris?")]


def weather_tool(seconds: float = 0.0, result: str | None = None) -> Tool:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        time.sleep(seconds)
        return result if result is not None else f"Sunny, 21C in {city}"

    return get_weather


def async_weather_tool(seconds: float, started: asyncio.Event | None = None) -> Tool:
    @tool
    async def get_weather(city: str) -> str:
        """Get the weather for a city, slowly."""
        if started is not None:
            started.set()
        await asyncio.sleep(seconds)
        return f"Sunny, 21C in {city}"

    return get_weather


def weather_agent(tool_seconds: float = 0.0, *, city: str = "Paris", result: str | None = None) -> Agent:
    return Agent(
        chat_generator=ScriptedChatGenerator(weather_script(city=city)), tools=[weather_tool(tool_seconds, result)]
    )


def failing_agent() -> Agent:
    error = RuntimeError("the provider's connection broke")
    turn = Turn(text_turn(["Hel", "lo"]).chunks, ChatMessage.from_assistant("Hello"), error=error, fail_after=1)
    return Agent(chat_generator=ScriptedChatGenerator([turn]))


def in_order_once(received: Subscription) -> list[str]:
    """The frames' data, after checking that their ids run from 1 without a gap or a repeat."""
    assert [frame.id for frame in received.frames] == [str(n) for n in range(1, len(received.frames) + 1)]
    return received.data


def recording(protocol: str, received: Subscription) -> dict[str, Any]:
    """A subscription as the front-end check replays it."""
    assert received.terminal is not None
    return {
        "protocol": protocol,
        "frames": [{"id": frame.id, "event": frame.event, "data": frame.data} for frame in received.frames],
        "terminal": {"event": received.terminal.event, "data": received.terminal.data},
    }


def types(received: Subscription) -> list[str]:
    return [json.loads(item)["type"] for item in received.data]


#: What the front-end checks need from the JavaScript checkout's node_modules.
JS_NEEDED = (
    "@neuron-core/iq-stream/dist/ag-ui",
    "@neuron-core/iq-stream/dist/ai-sdk",
    "@ag-ui/client",
    "@ag-ui/core",
    "ai",
)


def js_missing() -> str | None:
    """Why the front-end checks cannot run, or None: they need Node and IQ_STREAM_JS_DIR, a built checkout of
    neuron-core/iq-stream-js with its node_modules."""
    js_dir = os.environ.get("IQ_STREAM_JS_DIR", "")
    if not js_dir:
        return "IQ_STREAM_JS_DIR is not set: point it at a built checkout of neuron-core/iq-stream-js"
    for package in JS_NEEDED:
        if not (Path(js_dir) / "node_modules" / package).exists():
            return f"{package} is missing from {js_dir}/node_modules: run npm install and the build there"
    if shutil.which("node") is None:
        return "node is not on the PATH"
    return None
