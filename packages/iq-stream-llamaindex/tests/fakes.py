"""Fake models and tools shared by both suites: no API key, no network. No ``from __future__ import annotations``
here: LlamaIndex finds a tool's ``Context`` parameter by its annotation at run time.

``FakeStreamingLLM`` is the prototype's model: a subclass of LlamaIndex's own ``MockFunctionCallingLLM`` that streams a
reasoning delta, text deltas and, on its first call, one tool call in the final chunk, as providers do. ``ScriptedLLM``
answers each call with the next scripted assistant message, streamed as one chunk.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Any

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
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
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context, HumanResponseEvent, InputRequiredEvent
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
    return f"21 °C and sunny in {city}"


def slow_weather(seconds: float) -> Callable[..., Awaitable[str]]:
    """A weather tool, named ``slow_weather``, that stays silent for ``seconds``."""

    async def slow_weather(city: str) -> str:
        """Get the weather for a city, slowly."""
        await asyncio.sleep(seconds)
        return f"21 °C and sunny in {city}"

    return slow_weather


async def broken_weather(city: str) -> str:
    """Get the weather for a city."""
    raise ConnectionError(f"weather service down for {city}: token=sk-tool-secret")


def weather_agent(
    *, tool: Callable[..., Awaitable[str]] = get_weather, timeout: float | None = None, **llm: Any
) -> FunctionAgent:
    """The prototype's agent: one tool, the fake streaming model, a system prompt the browser must never see. The
    model calls the tool by its name."""
    return FunctionAgent(
        name="weather_agent",
        description="Answers weather questions",
        tools=[tool],
        system_prompt=SYSTEM_PROMPT,
        llm=FakeStreamingLLM(tool_name=tool.__name__, **llm),
        timeout=timeout,
    )


async def ask(ctx: Context, question: str) -> str:
    """Ask the human a question."""
    answer = await ctx.wait_for_event(
        HumanResponseEvent, waiter_id=question, waiter_event=InputRequiredEvent(prefix=question), timeout=30
    )
    return str(answer.response)


def hitl_agent() -> FunctionAgent:
    """An agent whose one tool waits for a human's answer before the model writes "Done."."""
    llm = ScriptedLLM(script=[[tool_call("ask", "q1", question="Proceed?")], [TextBlock(text="Done.")]])
    return FunctionAgent(name="assistant", tools=[ask], llm=llm)


BIG_RESULT = "x" * 200_000


async def big_weather(city: str) -> str:
    """Get a very detailed weather report for a city."""
    return BIG_RESULT


class Opaque:
    """A tool's raw output that JSON has no type for."""

    def __str__(self) -> str:
        return "an opaque report"


async def opaque_weather(city: str) -> Opaque:
    """Get the weather for a city as an object."""
    return Opaque()


def parallel_agent() -> FunctionAgent:
    """One model step that calls two tools at once; the slower one finishes last."""
    llm = ScriptedLLM(
        script=[
            [tool_call("slow_weather", "call_a", city="Rome"), tool_call("get_weather", "call_b", city="Oslo")],
            [TextBlock(text="Rome and Oslo are sunny.")],
        ]
    )
    return FunctionAgent(name="assistant", tools=[slow_weather(0.2), get_weather], llm=llm)


def handoff_workflow() -> AgentWorkflow:
    """Two agents in one workflow: triage hands the question to weather, which calls its tool and answers."""
    triage = FunctionAgent(
        name="triage",
        description="Routes questions",
        tools=[],
        can_handoff_to=["weather"],
        llm=ScriptedLLM(script=[[tool_call("handoff", "h1", to_agent="weather", reason="a weather question")]]),
    )
    weather = FunctionAgent(
        name="weather",
        description="Answers weather questions",
        tools=[get_weather],
        llm=ScriptedLLM(script=[[tool_call("get_weather", "t1", city="Rome")], [TextBlock(text="Sunny in Rome.")]]),
    )
    return AgentWorkflow(agents=[triage, weather], root_agent="triage")


REACT_ANSWER = "It is sunny in Paris."


def react_agent(*, streaming: bool = False) -> ReActAgent:
    """A ReActAgent: one tool step, then its answer, which llama-index-core strips of "Answer:" once written."""
    llm = ScriptedLLM(
        script=[
            [TextBlock(text='Thought: I need the weather.\nAction: get_weather\nAction Input: {"city": "Paris"}')],
            [TextBlock(text=f"Thought: I can answer.\nAnswer: {REACT_ANSWER}")],
        ]
    )
    return ReActAgent(name="react", tools=[get_weather], llm=llm, streaming=streaming)


def direct_agent() -> FunctionAgent:
    """An agent whose tool has return_direct=True: its result ends the run as the answer."""
    tool = FunctionTool.from_defaults(async_fn=get_weather, return_direct=True)
    return FunctionAgent(name="direct", tools=[tool], llm=FakeStreamingLLM(tool_name="get_weather"))


def repeated_id_agent(*, parallel: bool = False) -> FunctionAgent:
    """A model without tool-call ids, as Ollama's integration is: the id is the tool's name. It calls one tool twice,
    in two steps, or twice at once in one step, where both calls are open together and the second ends first."""

    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        await asyncio.sleep(0.2 if city == "Paris" else 0.05)
        return f"21 °C and sunny in {city}"

    paris = tool_call("get_weather", "get_weather", city="Paris")
    rome = tool_call("get_weather", "get_weather", city="Rome")
    calls = [[paris, rome]] if parallel else [[paris], [rome]]
    llm = ScriptedLLM(script=[*calls, [TextBlock(text="Paris and Rome are sunny.")]])
    return FunctionAgent(name="assistant", tools=[get_weather], llm=llm)
