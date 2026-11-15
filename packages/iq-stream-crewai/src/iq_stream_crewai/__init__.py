"""Publish a CrewAI run to browsers through an IQ-Stream server: everything CrewAI emits inside a ``with publish(...)``
block goes to one stream, as AG-UI events or as CrewAI's own."""

from __future__ import annotations

from collections.abc import Collection

from crewai import Crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llms.base_llm import BaseLLM

from ._encoder import Encoder
from ._limits import Frame
from ._snapshot import EVENT_FIELDS
from ._stream import CrewAIStream, publish

#: The CrewAI event types the adapter publishes, the default of ``include``.
EVENT_TYPES: Collection[str] = frozenset(EVENT_FIELDS)

__all__ = ["EVENT_TYPES", "CrewAIStream", "Encoder", "Frame", "enable_token_streaming", "publish"]


def enable_token_streaming(*runnables: Crew | BaseAgent | BaseLLM) -> None:
    """Turns streaming on for the models of these crews, agents and models, as ``Crew(stream=True)`` does inside its
    kickoff. Without it (or ``LLM(stream=True)``) a run publishes each answer as one block when its call returns."""
    for runnable in runnables:
        if isinstance(runnable, BaseLLM):
            runnable.stream = True
        elif isinstance(runnable, Crew):
            agents = [*runnable.agents, runnable.manager_agent]
            enable_token_streaming(*(agent for agent in agents if agent is not None))
            if isinstance(runnable.manager_llm, BaseLLM):
                runnable.manager_llm.stream = True
        elif isinstance(runnable, BaseAgent):  # pyright: ignore[reportUnnecessaryIsInstance]
            if isinstance(runnable.llm, BaseLLM):
                runnable.llm.stream = True
        else:
            raise TypeError(f"enable_token_streaming() takes crews, agents and models, not {type(runnable).__name__}.")
