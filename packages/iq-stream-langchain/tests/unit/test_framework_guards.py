"""Guards for the framework behaviour the design rests on: if one of these fails after an upgrade, the reason for
wrapping the iterator instead of using callbacks has changed."""

from __future__ import annotations

from typing import Any

from fakes import USER, weather_agent
from langchain_core.callbacks import BaseCallbackHandler


class TokenCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens += 1


def test_a_plain_callback_handler_on_invoke_gets_no_tokens() -> None:
    counter = TokenCounter()
    weather_agent().invoke(USER, config={"callbacks": [counter]})
    assert counter.tokens == 0, "a callback handler alone does not make the model stream"

    streamed = TokenCounter()
    for _ in weather_agent().stream(USER, config={"callbacks": [streamed]}, stream_mode="messages"):
        pass
    assert streamed.tokens > 0


def test_the_tools_mode_yields_tool_events() -> None:
    parts = list(weather_agent().stream(USER, stream_mode=["tools"], subgraphs=True, version="v2"))  # type: ignore[list-item]
    assert [part["data"]["event"] for part in parts] == ["tool-started", "tool-finished"]  # type: ignore[index]


def test_an_interrupted_run_ends_the_iterator_normally() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    agent = weather_agent(checkpointer=InMemorySaver())
    parts = list(
        agent.stream(
            USER,
            {"configurable": {"thread_id": "guard"}},
            stream_mode=["updates"],
            subgraphs=True,
            version="v2",
            interrupt_before=["tools"],
        )
    )
    assert "__interrupt__" in parts[-1]["data"]
