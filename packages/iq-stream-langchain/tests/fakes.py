"""A scripted chat model and small graphs for the tests: no API key, no network.

`FakeToolCallingModel` is LangChain's `GenericFakeChatModel` with the two things `create_agent` needs and the stock
fake lacks: `bind_tools`, and a stream that carries `tool_call_chunks`. Text streams word by word, a tool call as a
header and its arguments in two halves, the way a provider sends them.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import operator
import sys
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from functools import reduce
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, message_chunk_to_message
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

#: LangChain hands a run's callbacks to a model that a node calls without its config (create_agent's model node does)
#: in async code only from Python 3.11, whose asyncio.create_task takes a context. On 3.10 an async run streams whole
#: messages, not tokens.
ASYNC_TOKENS = sys.version_info >= (3, 11)

USER = {"messages": [{"role": "user", "content": "weather in Turin?"}]}
FIRST = "Let me check the weather."
FINAL = "It is sunny in Turin, 21 degrees."
TOOL_RESULT = "sunny, 21C in Turin"


class FakeToolCallingModel(GenericFakeChatModel):
    token_delay: float = 0.0
    #: Raise a RuntimeError once this many text chunks were streamed.
    fail_after_tokens: int | None = None

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeToolCallingModel:  # type: ignore[override]
        return self

    def _chunks(self) -> Iterator[ChatGenerationChunk]:
        message = next(self.messages)
        if isinstance(message, str):
            message = AIMessage(content=message)
        if not isinstance(message.content, str):  # content blocks: one chunk
            yield ChatGenerationChunk(message=AIMessageChunk(content=message.content, id=message.id))
        text = message.content if isinstance(message.content, str) else ""
        pieces = [piece for piece in text.replace(" ", " \x00").split("\x00") if piece]
        for sent, piece in enumerate(pieces):
            if self.fail_after_tokens is not None and sent >= self.fail_after_tokens:
                raise RuntimeError("fake model exploded mid-stream")
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece, id=message.id))
        for index, call in enumerate(message.tool_calls):
            # A header with the name and id, then the arguments in two fragments, as OpenAI and Anthropic send them.
            arguments = json.dumps(call["args"])
            half = len(arguments) // 2
            for position, part in enumerate(("", arguments[:half], arguments[half:])):
                chunk = AIMessageChunk(
                    content="",
                    id=message.id,
                    tool_call_chunks=[
                        {
                            "type": "tool_call_chunk",
                            "name": call["name"] if position == 0 else None,
                            "args": part,
                            "id": call["id"] if position == 0 else None,
                            "index": index,
                        }
                    ],
                )
                yield ChatGenerationChunk(message=chunk)
        yield ChatGenerationChunk(message=AIMessageChunk(content="", id=message.id, chunk_position="last"))

    def _merged(self, chunks: list[ChatGenerationChunk]) -> ChatResult:
        merged = reduce(operator.add, (chunk.message for chunk in chunks))
        return ChatResult(generations=[ChatGeneration(message=message_chunk_to_message(merged))])

    # Without streaming (no callback asked for tokens), the same message, the same failure, the same time.
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks: list[ChatGenerationChunk] = []
        for chunk in self._chunks():
            chunks.append(chunk)
            time.sleep(self.token_delay)
        return self._merged(chunks)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        chunks: list[ChatGenerationChunk] = []
        for chunk in self._chunks():
            chunks.append(chunk)
            await asyncio.sleep(self.token_delay)
        return self._merged(chunks)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._chunks():
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
            if self.token_delay:
                time.sleep(self.token_delay)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._chunks():
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
            if self.token_delay:
                await asyncio.sleep(self.token_delay)


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def scripted_model(*, repeat: bool = False, **options: Any) -> FakeToolCallingModel:
    """Turn 1: some text and a tool call. Turn 2: the final answer. With `repeat`, the same for every run."""
    turns = [
        AIMessage(content=FIRST, tool_calls=[tool_call("get_weather", {"city": "Turin"}, "call_1")]),  # type: ignore[list-item]
        AIMessage(content=FINAL),
    ]
    return FakeToolCallingModel(messages=itertools.cycle(turns) if repeat else iter(turns), **options)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny, 21C in {city}"


def weather_agent(**options: Any) -> Any:
    """A `create_agent` agent with one tool call. Options go to `create_agent`, except the model's own."""
    model_options = {key: options.pop(key) for key in ("token_delay", "fail_after_tokens") if key in options}
    return create_agent(scripted_model(**model_options), [get_weather], **options)


@tool
def book(city: str) -> str:
    """Book a trip; a human approves it first."""
    approved = interrupt({"question": f"book {city}?"})
    return f"booked {city}: {approved}"


def booking_agent(**options: Any) -> Any:
    """A `create_agent` agent whose tool waits for a human; with a checkpointer, `Command(resume=...)` resumes it."""
    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("book", {"city": "Turin"}, "c1")]), "Booked."]  # type: ignore[list-item]
        )
    )
    return create_agent(model, [book], **options)


def research_supervisor() -> Any:
    """A supervisor whose tool runs a `create_agent` sub-agent, which writes slowly."""
    researcher = create_agent(
        FakeToolCallingModel(
            messages=iter([AIMessage(content="the researcher writes a long and slow answer here")]), token_delay=0.1
        ),
        [],
        name="researcher",
    )

    @tool
    async def research(topic: str, config: RunnableConfig) -> str:
        """Ask the researcher."""
        # The config carries the run's callbacks to the sub-agent: on Python 3.10 nothing else does.
        answer = await researcher.ainvoke({"messages": [{"role": "user", "content": topic}]}, config)
        return str(answer["messages"][-1].text)

    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("research", {"topic": "Turin"}, "c1")]), "done"]  # type: ignore[list-item]
        )
    )
    return create_agent(model, [research], checkpointer=InMemorySaver(), name="supervisor")


@tool
def send_email(to: str, body: str) -> str:
    """Send an email; a human approves it first."""
    return f"sent to {to}"


#: A draft long enough that HumanInTheLoopMiddleware's interrupt (which repeats the arguments in its description)
#: does not fit the short server's 1 KiB frame, while each half of the streamed arguments does.
DRAFT = "Dear team, " + "the quarterly figures are attached. " * 12


def hitl_agent(body: str = DRAFT) -> Any:
    """A `create_agent` agent whose `send_email` call waits for a human's decision (HumanInTheLoopMiddleware)."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    model = FakeToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="", tool_calls=[tool_call("send_email", {"to": "team@example.com", "body": body}, "c1")]
                ),  # type: ignore[list-item]
                "Sent.",
            ]
        )
    )
    middleware = HumanInTheLoopMiddleware(interrupt_on={"send_email": True})
    return create_agent(model, [send_email], middleware=[middleware], checkpointer=InMemorySaver())


def stubborn_agent(cleanup: float, entered: asyncio.Event) -> Any:
    """A tool that finishes its work before it lets a cancel through (a transaction it must not leave half done).
    ``entered`` is set once the tool runs."""

    @tool
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        entered.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.sleep(cleanup))
            raise
        return "sunny"

    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Sunny."])  # type: ignore[list-item]
    )
    return create_agent(model, [get_weather], checkpointer=InMemorySaver())


def failing_agent(**options: Any) -> Any:
    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="This answer will never be finished at all")]), fail_after_tokens=3
    )
    return create_agent(model, [get_weather], **options)


class ParallelState(TypedDict):
    messages: Annotated[list[Any], operator.add]


def two_sub_agents(*, async_nodes: bool, delay_a: float = 0.03, delay_b: float = 0.02) -> Any:
    """Two `create_agent` sub-agents run by two nodes of one superstep: their chunks interleave."""

    def sub_agent(name: str, delay: float) -> Any:
        model = FakeToolCallingModel(
            messages=iter([AIMessage(content=f"{name} says one two three four five")]), token_delay=delay
        )
        return create_agent(model, [], name=name)

    alpha, beta = sub_agent("alpha", delay_a), sub_agent("beta", delay_b)
    go = {"messages": [{"role": "user", "content": "go"}]}

    def call_alpha(state: ParallelState) -> dict[str, Any]:
        return {"messages": [alpha.invoke(go)["messages"][-1]]}

    def call_beta(state: ParallelState) -> dict[str, Any]:
        return {"messages": [beta.invoke(go)["messages"][-1]]}

    async def acall_alpha(state: ParallelState) -> dict[str, Any]:
        return {"messages": [(await alpha.ainvoke(go))["messages"][-1]]}

    async def acall_beta(state: ParallelState) -> dict[str, Any]:
        return {"messages": [(await beta.ainvoke(go))["messages"][-1]]}

    graph = StateGraph(ParallelState)
    graph.add_node("a", acall_alpha if async_nodes else call_alpha)
    graph.add_node("b", acall_beta if async_nodes else call_beta)
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_edge("a", END)
    graph.add_edge("b", END)
    return graph.compile()


def text_of(parts: Sequence[dict[str, Any]], ns_prefix: str | None = None) -> str:
    """The assistant text that `messages` parts (decoded frames) carry, in order."""
    return "".join(
        part["data"][0]["content"]
        for part in parts
        if part["type"] == "messages"
        and part["data"][0]["type"] in ("AIMessageChunk", "ai")  # tokens, or a whole message (async on 3.10)
        and (ns_prefix is None or (part["ns"] and part["ns"][0].startswith(ns_prefix)))
    )
