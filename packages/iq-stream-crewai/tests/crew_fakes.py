"""Fake models and a crew for the tests: no network, no API key.

The models subclass ``crewai.llms.base_llm.BaseLLM``, as CrewAI's own tests do, and emit the events a real provider
emits (``crewai/llms/providers/openai/completion.py``): ``LLMCallStartedEvent``, one ``LLMStreamChunkEvent`` per
delta when streaming is on (text, or a tool call whose ``chunk`` is the argument delta and whose
``tool_call.function.arguments`` is the accumulated string), ``LLMThinkingChunkEvent`` for reasoning, then
``LLMCallCompletedEvent`` or ``LLMCallFailedEvent``.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from crewai import Agent, Crew, Task
from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.tools import BaseTool
from pydantic import Field

ANSWER = "The capital of France is Paris."
TERM = "capital of France"


class FakeStreamingLLM(BaseLLM):
    """First call: one native tool call of ``lookup`` per entry of ``tool_args`` (several are run in parallel by
    CrewAI). Second call: the answer, word by word. ``fail_on_call`` raises on that call after one chunk."""

    answer: str = ANSWER
    tool_args: list[dict[str, Any]] = Field(default_factory=lambda: [])
    delay: float = 0.0
    fail_on_call: int | None = None
    thinking: str | None = None
    #: Tool call ids that do not change from call to call, as some models number them from zero in each response.
    fixed_ids: bool = False
    calls: int = 0

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "fake-stream")
        data.setdefault("stream", True)
        super().__init__(**data)

    def supports_function_calling(self) -> bool:
        return True

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        self.calls += 1
        number = self.calls
        streaming = bool(self._effective_stream())
        where = {"from_task": from_task, "from_agent": from_agent}
        with llm_call_context() as call_id:
            self._emit_call_started_event(messages=messages, tools=tools, **where)
            response_id = f"resp_{call_id}"
            if self.fail_on_call == number:
                if streaming:
                    self._emit_stream_chunk_event(chunk="Let me ", response_id=response_id, **where)
                self._emit_call_failed_event(error="fake provider outage at sk-secret", **where)
                raise RuntimeError("fake provider outage at sk-secret")
            if number == 1 and tools and self.tool_args:
                calls: list[dict[str, Any]] = []
                for index, spec in enumerate(self.tool_args):
                    tool_call_id = f"call_{index}" if self.fixed_ids else f"call_{call_id[:8]}_{index}"
                    args = json.dumps(spec)
                    if streaming:
                        accumulated = ""
                        # The first chunk announces the call (id and name, no arguments), as Anthropic's
                        # content_block_start does; the next ones carry argument deltas.
                        for delta in ["", *[args[i : i + 6] for i in range(0, len(args), 6)]]:
                            accumulated += delta
                            time.sleep(self.delay)
                            self._emit_stream_chunk_event(
                                chunk=delta,
                                tool_call={
                                    "id": tool_call_id,
                                    "type": "function",
                                    "index": index,
                                    "function": {"name": "lookup", "arguments": accumulated},
                                },
                                call_type=LLMCallType.TOOL_CALL,
                                response_id=response_id,
                                **where,
                            )
                    calls.append(
                        {"id": tool_call_id, "type": "function", "function": {"name": "lookup", "arguments": args}}
                    )
                self._emit_call_completed_event(
                    response=calls,
                    call_type=LLMCallType.TOOL_CALL,
                    messages=messages,
                    finish_reason="tool_calls",
                    response_id=response_id,
                    **where,
                )
                return calls
            if streaming and self.thinking:
                for piece in self.thinking.split(" "):
                    time.sleep(self.delay)
                    self._emit_thinking_chunk_event(chunk=piece + " ", response_id=response_id, **where)
            if streaming:
                words = self.answer.split(" ")
                for i, word in enumerate(words):
                    time.sleep(self.delay)
                    chunk = word + (" " if i < len(words) - 1 else "")
                    self._emit_stream_chunk_event(chunk=chunk, response_id=response_id, **where)
            self._emit_call_completed_event(
                response=self.answer,
                call_type=LLMCallType.LLM_CALL,
                messages=messages,
                finish_reason="stop",
                response_id=response_id,
                **where,
            )
            return self.answer

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)


class FakeReActLLM(BaseLLM):
    """A model without native function calling: CrewAI's text protocol (``Action:``, ``Final Answer:``)."""

    calls: int = 0

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "fake-react")
        data.setdefault("stream", True)
        super().__init__(**data)

    def supports_function_calling(self) -> bool:
        return False

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        self.calls += 1
        if self.calls == 1:
            text = 'Thought: I should look it up.\nAction: lookup\nAction Input: {"term": "x"}'
        else:
            text = "Thought: I now know the final answer\nFinal Answer: It is value:x."
        where = {"from_task": from_task, "from_agent": from_agent}
        with llm_call_context():
            self._emit_call_started_event(messages=messages, tools=tools, **where)
            for i in range(0, len(text), 12):
                self._emit_stream_chunk_event(chunk=text[i : i + 12], **where)
            self._emit_call_completed_event(response=text, call_type=LLMCallType.LLM_CALL, messages=messages, **where)
            return text

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."
    seconds: float = 0.0
    explode: bool = False
    #: When set, the tool waits for it (at most ``seconds``) instead of sleeping.
    gate: Any = None
    #: When set, the tool returns this instead of ``value:<term>``.
    result: str | None = None

    def _run(self, term: str = "default") -> str:
        if isinstance(self.gate, threading.Event):
            self.gate.wait(self.seconds)
        elif self.seconds:
            time.sleep(self.seconds)
        if self.explode:
            raise RuntimeError("lookup backend is down at /srv/secret")
        return self.result if self.result is not None else f"value:{term}"


def make_crew(
    *,
    name: str = "demo",
    answer: str = ANSWER,
    terms: list[str] | None = None,
    tool: LookupTool | None = None,
    fail_on_call: int | None = None,
    delay: float = 0.0,
    stream: bool = True,
    crew_stream: bool = False,
    thinking: str | None = None,
    llm: BaseLLM | None = None,
) -> Crew:
    """One agent with the ``lookup`` tool and one task. ``terms``: one tool call per term (``[TERM]`` by default;
    ``[]`` for none)."""
    terms = [TERM] if terms is None else terms
    model = llm or FakeStreamingLLM(
        answer=answer,
        tool_args=[{"term": term} for term in terms],
        delay=delay,
        fail_on_call=fail_on_call,
        stream=stream,
        thinking=thinking,
    )
    agent = Agent(
        role=f"Researcher {name}",
        goal="Answer the question",
        backstory="A test agent",
        llm=model,
        tools=[tool or LookupTool()],
        max_retry_limit=0,
        verbose=False,
    )
    task = Task(description=f"Answer question {name}", expected_output="One sentence", agent=agent)
    return Crew(name=f"crew-{name}", agents=[agent], tasks=[task], verbose=False, stream=crew_stream)
