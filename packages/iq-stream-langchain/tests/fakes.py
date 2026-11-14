"""A scripted chat model and small graphs for the tests: no API key, no network.

`FakeToolCallingModel` is LangChain's `GenericFakeChatModel` with the two things `create_agent` needs and the stock
fake lacks: `bind_tools`, and a stream that carries `tool_call_chunks`. Text streams word by word, a tool call's
arguments in two halves, the way a provider sends them.
"""

from __future__ import annotations

import asyncio
import json
import operator
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

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
        text = message.content if isinstance(message.content, str) else ""
        pieces = [piece for piece in text.replace(" ", " \x00").split("\x00") if piece]
        for sent, piece in enumerate(pieces):
            if self.fail_after_tokens is not None and sent >= self.fail_after_tokens:
                raise RuntimeError("fake model exploded mid-stream")
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece, id=message.id))
        for index, call in enumerate(message.tool_calls):
            arguments = json.dumps(call["args"])
            half = len(arguments) // 2
            for position, part in enumerate((arguments[:half], arguments[half:])):
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


def scripted_model(**options: Any) -> FakeToolCallingModel:
    """Turn 1: some text and a tool call. Turn 2: the final answer."""
    return FakeToolCallingModel(
        messages=iter(
            [
                AIMessage(content=FIRST, tool_calls=[tool_call("get_weather", {"city": "Turin"}, "call_1")]),  # type: ignore[list-item]
                AIMessage(content=FINAL),
            ]
        ),
        **options,
    )


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"sunny, 21C in {city}"


def weather_agent(**options: Any) -> Any:
    """A `create_agent` agent with one tool call. Options go to `create_agent`, except the model's own."""
    model_options = {key: options.pop(key) for key in ("token_delay", "fail_after_tokens") if key in options}
    return create_agent(scripted_model(**model_options), [get_weather], **options)


def failing_agent(**options: Any) -> Any:
    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="This answer will never be finished at all")]), fail_after_tokens=3
    )
    return create_agent(model, [get_weather], **options)


class ParallelState(TypedDict):
    messages: Annotated[list[Any], operator.add]


def two_sub_agents(delay_a: float = 0.03, delay_b: float = 0.02) -> Any:
    """Two `create_agent` sub-agents run by two nodes of one superstep: their chunks interleave."""

    def sub_agent(name: str, delay: float) -> Any:
        model = FakeToolCallingModel(
            messages=iter([AIMessage(content=f"{name} says one two three four five")]), token_delay=delay
        )
        return create_agent(model, [], name=name)

    alpha, beta = sub_agent("alpha", delay_a), sub_agent("beta", delay_b)

    async def call_alpha(state: ParallelState) -> dict[str, Any]:
        result = await alpha.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        return {"messages": [result["messages"][-1]]}

    async def call_beta(state: ParallelState) -> dict[str, Any]:
        result = await beta.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        return {"messages": [result["messages"][-1]]}

    graph = StateGraph(ParallelState)
    graph.add_node("a", call_alpha)
    graph.add_node("b", call_beta)
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
        and part["data"][0]["type"] == "AIMessageChunk"
        and (ns_prefix is None or (part["ns"] and part["ns"][0].startswith(ns_prefix)))
    )
