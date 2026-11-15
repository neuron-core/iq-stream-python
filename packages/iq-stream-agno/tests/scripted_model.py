"""A scripted Agno model, no network and no API key (ported from the plan's prototype, fake_model.py).

Turn 1 streams a few text deltas (unless ``preamble`` is off), then one or more tool calls; turn 2, once the tool
results are in, streams the answer. ``fail_on_turn`` makes a turn raise after its first deltas. Modelled on the
scripted models of Agno's own tests (libs/agno/tests/integration/os/test_approval_multi_pause.py), since Agno ships
none.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agno.agent import Agent
from agno.metrics import MessageMetrics
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.team import Team
from agno.tools import tool

ANSWER = "It is 21 degrees and sunny in Paris."


class ScriptedModel(Model):
    def __init__(
        self,
        tool_name: str | None = "get_weather",
        tool_args: dict[str, Any] | None = None,
        fail_on_turn: int | None = None,
        delay: float = 0.005,
        reasoning: bool = False,
        answer: str = ANSWER,
        n_tool_calls: int = 1,
        preamble: bool = True,
    ) -> None:
        super().__init__(id="scripted", name="scripted", provider="test")
        self.tool_name = tool_name
        self.tool_args = tool_args or {"city": "Paris"}
        self.fail_on_turn = fail_on_turn
        self.delay = delay
        self.reasoning = reasoning
        self.answer = answer
        self.n_tool_calls = n_tool_calls
        self.preamble = preamble

    def _script(self, messages: list[Any]) -> list[ModelResponse | Exception]:
        last_user = max((i for i, message in enumerate(messages) if message.role == "user"), default=-1)
        answered = sum(1 for message in messages[last_user + 1 :] if message.role == "tool")
        turn = 1 if answered == 0 else 2
        out: list[ModelResponse | Exception] = []
        if turn == 1 and self.tool_name:
            if self.reasoning:
                for part in ["The user wants weather. ", "I should call the tool."]:
                    out.append(ModelResponse(role="assistant", reasoning_content=part))
            for part in ["Let me ", "check ", "the weather. "] if self.preamble else []:
                out.append(ModelResponse(role="assistant", content=part))
            call = ModelResponse(role="assistant")
            call.tool_calls = [
                {
                    "id": f"call_{i + 1}",
                    "type": "function",
                    "function": {"name": self.tool_name, "arguments": json.dumps(self.tool_args)},
                }
                for i in range(self.n_tool_calls)
            ]
            out.append(call)
        else:
            words = self.answer.split(" ")
            for i, word in enumerate(words):
                out.append(ModelResponse(role="assistant", content=word + ("" if i == len(words) - 1 else " ")))
        last = out[-1]
        assert isinstance(last, ModelResponse)
        last.response_usage = MessageMetrics(input_tokens=10, output_tokens=5, total_tokens=15)
        if self.fail_on_turn == turn:
            out = [*out[:2], RuntimeError("scripted model failure")]
        return out

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        parts = [part for part in self._script(kwargs["messages"]) if isinstance(part, ModelResponse)]
        response = ModelResponse(role="assistant", content="".join(part.content or "" for part in parts))
        for part in parts:
            if part.tool_calls:
                response.tool_calls = part.tool_calls
        return response

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self.invoke(*args, **kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        for part in self._script(kwargs["messages"]):
            time.sleep(self.delay)
            if isinstance(part, Exception):
                raise part
            yield part

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        for part in self._script(kwargs["messages"]):
            await asyncio.sleep(self.delay)
            if isinstance(part, Exception):
                raise part
            yield part

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response if isinstance(response, ModelResponse) else ModelResponse()

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response if isinstance(response, ModelResponse) else ModelResponse()


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city (str): City name
    """
    return f"21C and sunny in {city}"


#: Seconds that ``slow_weather`` takes; a test sets it.
SLOW = {"seconds": 0.3}


@tool
def slow_weather(city: str) -> str:
    """Get the weather for a city, slowly.

    Args:
        city (str): City name
    """
    time.sleep(SLOW["seconds"])
    return f"21C and sunny in {city}"


#: The size of ``big_report``'s result; a test sets it.
BIG = {"size": 100_000}


@tool
def big_report(city: str) -> str:
    """Return a large report.

    Args:
        city (str): City name
    """
    return "x" * BIG["size"]


@tool
def broken_tool(city: str) -> str:
    """A tool that fails.

    Args:
        city (str): City name
    """
    raise RuntimeError("secret database password in this message")


@tool(requires_confirmation=True)
def book_trip(city: str) -> str:
    """Book a trip.

    Args:
        city (str): City name
    """
    return f"booked {city}"


@tool(requires_confirmation=True)
def slow_book(city: str) -> str:
    """Book a trip, slowly.

    Args:
        city (str): City name
    """
    time.sleep(SLOW["seconds"])
    return f"booked {city}"


def weather_agent(**model_options: Any) -> Agent:
    """The prototype's agent: one tool call, then the answer."""
    tool_name = model_options.get("tool_name", "get_weather")
    tools = {
        "get_weather": get_weather,
        "slow_weather": slow_weather,
        "big_report": big_report,
        "broken_tool": broken_tool,
        "book_trip": book_trip,
        "slow_book": slow_book,
    }
    extra: dict[str, Any] = {}
    if "db" in model_options:
        extra["db"] = model_options.pop("db")
    return Agent(
        id="weather-agent",
        name="Weather",
        model=ScriptedModel(**model_options),
        tools=[tools[tool_name]] if tool_name else [],
        telemetry=False,
        **extra,
    )


def team(member_fails: bool = False, leader_fails: bool = False, preamble: bool = True) -> Team:
    """A leader that delegates to one member, then answers with what it said."""
    member = Agent(
        id="weather-member",
        name="Weather Member",
        role="weather",
        model=ScriptedModel(fail_on_turn=2 if member_fails else None),
        tools=[get_weather],
        telemetry=False,
    )
    leader = ScriptedModel(
        tool_name="delegate_task_to_member",
        tool_args={"member_id": "weather-member", "task": "weather in Paris"},
        answer="The member says it is sunny.",
        fail_on_turn=2 if leader_fails else None,
        preamble=preamble,
    )
    return Team(id="weather-team", name="Team", model=leader, members=[member], telemetry=False)
