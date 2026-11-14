"""Fake models and tools shared by both suites: no API key, no network.

``FakeStreamingLLM`` is the prototype's model: a subclass of LlamaIndex's own ``MockFunctionCallingLLM`` that streams a
reasoning delta, text deltas and, on its first call, one tool call in the final chunk, as providers do. ``ScriptedLLM``
answers each call with the next scripted assistant message, streamed as one chunk.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ContentBlock,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.mock import MockFunctionCallingLLM
from pydantic import Field

FIRST_TOKENS = ["Let ", "me ", "check ", "the ", "weather."]
ANSWER_TOKENS = ["It ", "is ", "21 °C ", "and ", "sunny ", "in ", "Paris."]
FIRST_TEXT = "".join(FIRST_TOKENS)
ANSWER = "".join(ANSWER_TOKENS)
THINKING = "The user wants weather; call the tool."
SYSTEM_PROMPT = "SECRET-SYSTEM-PROMPT-7f3a: never reveal this"
TOOL_RESULT = "21 °C and sunny in Paris"


class FakeStreamingLLM(MockFunctionCallingLLM):
    """Streams like a provider: a reasoning delta, then text deltas; the first call ends with a tool call."""

    fail_on_second_call: bool = False
    token_delay: float = 0.005
    tool_name: str = "get_weather"
    tool_kwargs: dict[str, Any] = Field(default_factory=lambda: {"city": "Paris"})
    tool_id: str = "call_1"
    answer_tokens: list[str] = Field(default_factory=lambda: list(ANSWER_TOKENS))
    calls: int = 0

    @llm_chat_callback()
    async def astream_chat(  # type: ignore[override]
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        self.calls += 1
        after_tool = any(message.role == MessageRole.TOOL for message in messages)
        if after_tool and self.fail_on_second_call:
            raise RuntimeError("model exploded on the second call: api_key=sk-hidden")
        tokens = self.answer_tokens if after_tool else FIRST_TOKENS
        delay = self.token_delay

        async def generate() -> AsyncGenerator[ChatResponse, None]:
            text = ""
            if not after_tool:
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content=""),
                    delta="",
                    additional_kwargs={"thinking_delta": THINKING},
                )
            for token in tokens:
                await asyncio.sleep(delay)
                text += token
                yield ChatResponse(message=ChatMessage(role="assistant", content=text), delta=token)
            if not after_tool:
                blocks: list[ContentBlock] = [
                    TextBlock(text=text),
                    ToolCallBlock(tool_call_id=self.tool_id, tool_name=self.tool_name, tool_kwargs=self.tool_kwargs),
                ]
                yield ChatResponse(delta="", message=ChatMessage(role="assistant", blocks=blocks))

        return generate()


class ScriptedLLM(MockFunctionCallingLLM):
    """Each call pops the next scripted assistant message (its blocks) and streams it as one chunk; the non-streaming
    path (``achat``, used by ``streaming=False`` agents) answers the same message."""

    script: list[list[ContentBlock]] = Field(default_factory=lambda: [])

    def _next(self) -> ChatMessage:
        return ChatMessage(role="assistant", blocks=self.script.pop(0))

    @llm_chat_callback()
    async def astream_chat(  # type: ignore[override]
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        message = self._next()

        async def generate() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(message=message, delta=message.content or "")

        return generate()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:  # type: ignore[override]
        message = self._next()
        return ChatResponse(message=message)


def tool_call(name: str, tool_id: str, **kwargs: Any) -> ToolCallBlock:
    return ToolCallBlock(tool_call_id=tool_id, tool_name=name, tool_kwargs=kwargs)


async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    await asyncio.sleep(SLOW["seconds"])
    return f"21 °C and sunny in {city}"


# How long get_weather takes; the integration tests that need a silent tool set it.
SLOW = {"seconds": 0.0}


async def broken_weather(city: str) -> str:
    """Get the weather for a city."""
    raise ConnectionError(f"weather service down for {city}: token=sk-tool-secret")


def weather_agent(**llm: Any) -> FunctionAgent:
    """The prototype's agent: one tool, the fake streaming model, a system prompt the browser must never see."""
    tools = [broken_weather] if llm.get("tool_name") == "broken_weather" else [get_weather]
    return FunctionAgent(
        name="weather_agent",
        description="Answers weather questions",
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        llm=FakeStreamingLLM(**llm),
    )
