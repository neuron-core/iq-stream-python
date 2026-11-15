"""What CrewAI does for itself is not the application's run: its internal flows, its own agents and tasks (a task's
guardrail, reasoning, crew planning, memory), and the model calls with no agent it makes around an agent's run
(memory's analysis, the conversion of an output, the summary of a full context window) are left out. Every other model
call is published, whoever made it: the agent's forced final answer, a model wrapper, a flow in a tool."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
from agui_checks import checked, conversation, types
from crew_fakes import ANSWER, TERM, FakeMemoryLLM, FakeStreamingLLM, LookupTool, PromptLLM, embed, make_crew
from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, start
from crewai.llms.base_llm import BaseLLM
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.unified_memory import Memory
from crewai.tools import BaseTool
from iq_stream.testing import RecordingPublisher
from pydantic import BaseModel

from iq_stream_crewai import EVENT_TYPES, publish
from iq_stream_crewai._limits import Fitter
from iq_stream_crewai._native import NativeEncoder
from iq_stream_crewai._snapshot import Snapshotter

Transport = Literal["sink", "bus"]
PROSE = "Thought: done\nFinal Answer: The city is Paris."
CITY = json.dumps({"city": "Paris"})


class City(BaseModel):
    city: str


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[Memory]:
    memory = Memory(llm=FakeMemoryLLM(), embedder=embed, storage=str(tmp_path / "memory"))
    yield memory
    memory.close()


def texts(publisher: RecordingPublisher) -> list[str]:
    return list(conversation(checked(publisher.data)).texts.values())


def roles(publisher: RecordingPublisher) -> set[str]:
    """The agents named by the frames of protocol="crewai"."""
    return {json.loads(data)["agent_role"] for data in publisher.data if '"agent_role"' in data}


# -- CrewAI's own work is left out -----------------------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["sink", "bus"])
def test_a_crew_s_memory_publishes_neither_its_model_calls_nor_its_flows(memory: Memory, transport: Transport) -> None:
    crew = make_crew()
    crew.memory = memory
    publisher = RecordingPublisher()
    with publish(publisher, transport=transport) as run:
        crew.kickoff()
    memory_llm = memory._llm  # pyright: ignore[reportPrivateUsage]
    assert isinstance(memory_llm, FakeMemoryLLM) and memory_llm.calls >= 2  # extraction and analysis ran
    events = checked(publisher.data)
    talk = conversation(events)
    assert list(talk.texts.values()) == [ANSWER]  # the agent's answer, and no memory JSON after it
    assert types(events).count("TEXT_MESSAGE_START") == 1
    assert run.errors == []

    crew = make_crew()
    crew.memory = memory
    native = RecordingPublisher()
    with publish(native, protocol="crewai", transport=transport):
        crew.kickoff()
    names = [frame.event for frame in native.frames]
    assert not [name for name in names if name and name.startswith("flow_")]  # RecallFlow, EncodingFlow
    models = {json.loads(frame.data)["model"] for frame in native.frames if frame.event == "llm_call_started"}
    assert models == {"fake-stream"}
    assert names.count("llm_call_completed") == 2


def test_a_model_call_inside_memory_s_own_work_is_left_out_outside_its_flows(tmp_path: Path) -> None:
    # A storage that reranks with a model: memory's work, in memory's scope, though no internal flow runs.
    reranker = FakeStreamingLLM(answer="1, 2, 3")

    class Reranking:
        def search(self, *args: object, **kwargs: object) -> list[object]:
            reranker.call("Rank these")
            return []

    memory = Memory(llm=FakeMemoryLLM(), embedder=embed, storage=str(tmp_path / "memory"))
    memory._storage = Reranking()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    publisher = RecordingPublisher()
    try:
        with publish(publisher):
            memory.recall("Paris?", depth="shallow")
    finally:
        memory.close()
    assert reranker.calls == 1
    assert types(checked(publisher.data)) == ["RUN_STARTED", "RUN_FINISHED"]


def test_an_internal_flow_s_model_calls_are_left_out_even_when_they_stream() -> None:
    analysis = json.dumps({"keywords": ["paris"], "suggested_scopes": ["/"], "complexity": "simple"})

    class Storage:
        def search(self, *args: object, **kwargs: object) -> list[object]:
            return []

        def list_scopes(self, *args: object, **kwargs: object) -> list[str]:
            return ["/"]

        def get_scope_info(self, *args: object, **kwargs: object) -> None:
            return None

    llm = FakeStreamingLLM(answer=analysis)
    publisher = RecordingPublisher()
    with publish(publisher):
        # A query over the analysis threshold: the flow asks its model.
        RecallFlow(storage=Storage(), llm=llm, embedder=embed).kickoff(inputs={"query": "What about Paris? " * 20})
    assert llm.calls == 1
    assert types(checked(publisher.data)) == ["RUN_STARTED", "RUN_FINISHED"]


@pytest.mark.parametrize("transport", ["sink", "bus"])
def test_a_task_guardrail_s_agent_is_left_out(transport: Transport) -> None:
    # A string guardrail is checked by CrewAI's own "Guardrail Agent", whose verdict would end the conversation.
    verdict = json.dumps({"valid": True, "feedback": None})
    llm = PromptLLM(answers={"complies with the given guardrail": verdict})
    agent = Agent(role="Researcher", goal="Answer", backstory="b", llm=llm, verbose=False)

    def crew() -> Crew:
        task = Task(description="Answer", expected_output="One sentence", agent=agent, guardrail="One sentence only")
        return Crew(agents=[agent], tasks=[task], verbose=False)

    publisher = RecordingPublisher()
    with publish(publisher, transport=transport) as run:
        result = crew().kickoff()
    assert result.raw == ANSWER
    assert llm.calls == 2  # the answer, and the guardrail's verdict
    assert texts(publisher) == [ANSWER]
    assert conversation(checked(publisher.data)).text == ANSWER
    assert run.errors == []

    native = RecordingPublisher()
    with publish(native, protocol="crewai", transport=transport):
        crew().kickoff()
    assert roles(native) == {"Researcher"}
    assert [frame.event for frame in native.frames].count("llm_call_started") == 1


def test_crew_planning_is_left_out() -> None:
    plan = json.dumps({"list_of_plans_per_task": [{"task_number": 1, "task": "Answer", "plan": "Step 1: answer."}]})
    agent = Agent(role="Researcher", goal="Answer", backstory="b", llm=PromptLLM(), verbose=False)
    task = Task(description="Answer", expected_output="One sentence", agent=agent)
    planner = PromptLLM(answer=plan)
    crew = Crew(agents=[agent], tasks=[task], planning=True, planning_llm=planner, verbose=False)
    publisher = RecordingPublisher()
    with publish(publisher):
        crew.kickoff()
    assert planner.calls == 1
    events = checked(publisher.data)
    assert list(conversation(events).texts.values()) == [ANSWER]
    # The planner's task is not a step either; the crew's own task, which planning extends, is.
    [step] = [event["stepName"] for event in events if event["type"] == "STEP_STARTED"]
    assert step.startswith("task:Answer") and "Planner" not in step


def test_an_agent_s_plan_is_left_out() -> None:
    from crewai.agent.planning_config import PlanningConfig

    ready = "1. Answer from memory.\nREADY: I am ready to execute the task."
    llm = PromptLLM(answers={"NOT READY": ready})
    planning = PlanningConfig(max_attempts=1, reasoning_effort="low")
    agent = Agent(role="Researcher", goal="Answer", backstory="b", llm=llm, planning_config=planning, verbose=False)
    task = Task(description="Answer", expected_output="One sentence", agent=agent)
    publisher = RecordingPublisher()
    with publish(publisher):
        Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    assert llm.calls == 2
    assert texts(publisher) == [ANSWER]


class TwoStep(PromptLLM):
    """Answers in prose, then, asked to convert it, in JSON."""

    def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.answer = PROSE if self.calls == 0 else CITY
        return super().call(messages, *args, **kwargs)


@pytest.mark.parametrize("transport", ["sink", "bus"])
def test_the_conversion_of_a_task_s_output_is_left_out(transport: Transport) -> None:
    llm = TwoStep()
    agent = Agent(role="Researcher", goal="Answer", backstory="b", llm=llm, verbose=False)
    task = Task(description="Name the city", expected_output="A city", agent=agent, output_pydantic=City)
    publisher = RecordingPublisher()
    with publish(publisher, transport=transport):
        result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
    assert llm.calls == 2 and result.pydantic == City(city="Paris")
    assert texts(publisher) == [PROSE]


def test_the_conversion_of_an_agent_kickoff_s_output_is_left_out() -> None:
    llm = TwoStep()
    agent = Agent(role="Solo", goal="Answer", backstory="b", llm=llm, verbose=False)
    publisher = RecordingPublisher()
    with publish(publisher):
        result = agent.kickoff("Which city?", response_format=City)
    assert llm.calls == 2 and result.pydantic == City(city="Paris")
    assert texts(publisher) == [PROSE]


@pytest.mark.skipif(
    not hasattr(BaseLLM, "is_context_window_overflowing_threshold"),
    reason="this CrewAI summarizes a full context window without announcing it (CrewAI 1.15.2): the summary is "
    "published, as the README says",
)
def test_the_summary_of_a_full_context_window_is_left_out() -> None:
    summary = "<summary>A lookup of the capital.</summary>"

    class Full(FakeStreamingLLM):
        def is_context_window_overflowing_threshold(self, *args: Any, **kwargs: Any) -> bool:
            return self.calls == 1  # after the tool call, before the answer

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
            answer = self.answer
            if kwargs.get("from_agent") is None:
                self.answer = summary  # CrewAI's summary request, which names no agent
            try:
                return super().call(messages, *args, **kwargs)
            finally:
                self.answer = answer

    llm = Full(tool_args=[{"term": TERM}])
    publisher = RecordingPublisher()
    with publish(publisher) as run:
        result = make_crew(llm=llm).kickoff()
    assert llm.calls == 3 and result.raw == ANSWER
    assert texts(publisher) == [ANSWER]
    assert run.errors == []


def test_scopes_are_forgotten_when_they_close() -> None:
    snapshotter = Snapshotter(frozenset(EVENT_TYPES), None, False)
    publisher = RecordingPublisher()
    encoder = NativeEncoder(snapshotter, Fitter(publisher.max_frame_bytes, publisher.frame_size, lambda error: None))
    with publish(publisher, protocol=encoder):
        make_crew().kickoff()
        assert snapshotter._open == {}  # pyright: ignore[reportPrivateUsage]
        assert snapshotter._summarizing == set()  # pyright: ignore[reportPrivateUsage]
    assert publisher.frames[-1].event == "crew_kickoff_completed"


# -- everything else is published ------------------------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["sink", "bus"])
def test_the_final_answer_crewai_forces_at_max_iter_is_published(transport: Transport) -> None:
    # The tool call takes the only iteration; CrewAI then asks the agent's model for the answer, naming no agent.
    crew = make_crew()
    crew.agents[0].max_iter = 1
    publisher = RecordingPublisher()
    with publish(publisher, transport=transport) as run:
        result = crew.kickoff()
    assert result.raw == ANSWER
    talk = conversation(checked(publisher.data))
    assert list(talk.texts.values()) == [ANSWER]
    assert talk.text == ANSWER  # the conversation ends with the answer
    assert run.errors == []

    crew = make_crew()
    crew.agents[0].max_iter = 1
    native = RecordingPublisher()
    with publish(native, protocol="crewai", transport=transport):
        crew.kickoff()
    chunks = "".join(json.loads(f.data)["chunk"] for f in native.frames if f.event == "llm_stream_chunk")
    assert chunks.endswith(ANSWER)


def test_an_agent_kickoff_at_max_iter_publishes_the_forced_answer() -> None:
    agent = Agent(
        role="Solo",
        goal="Answer",
        backstory="b",
        llm=FakeStreamingLLM(tool_args=[{"term": TERM}]),
        tools=[LookupTool()],
        max_iter=1,
        verbose=False,
    )
    publisher = RecordingPublisher()
    with publish(publisher):
        agent.kickoff("What is the capital of France?")
    assert texts(publisher) == [ANSWER]


def test_a_model_wrapper_that_names_no_agent_is_published() -> None:
    # Retry, fallback or caching wrappers that do not pass from_agent on: CrewAI runs, and so does the stream.
    class Fallback(BaseLLM):
        primary: Any = None

        def __init__(self, **data: Any) -> None:
            data.setdefault("model", "fallback")
            super().__init__(**data)

        def supports_function_calling(self) -> bool:
            return True

        def call(self, messages: Any, tools: Any = None, callbacks: Any = None, **kwargs: Any) -> Any:
            return self.primary.call(messages, tools=tools, available_functions=kwargs.get("available_functions"))

    publisher = RecordingPublisher()
    with publish(publisher):
        result = make_crew(llm=Fallback(primary=FakeStreamingLLM(tool_args=[{"term": TERM}]))).kickoff()
    assert result.raw == ANSWER
    talk = conversation(checked(publisher.data))
    assert list(talk.texts.values()) == [ANSWER]
    assert list(talk.tool_results.values()) == [f"value:{TERM}"]


def test_the_application_s_own_model_calls_are_published() -> None:
    llm = FakeStreamingLLM(answer="Said directly.")

    class Direct(Flow):  # type: ignore[type-arg]
        @start()
        def ask(self) -> str:
            return llm.call("Say it")

    publisher = RecordingPublisher()
    with publish(publisher):
        llm.call("Say it")
        Direct().kickoff()
    events = checked(publisher.data)
    assert list(conversation(events).texts.values()) == ["Said directly.", "Said directly."]


@pytest.mark.parametrize("transport", ["sink", "bus"])
def test_the_application_s_flow_and_model_in_a_tool_are_published(transport: Transport) -> None:
    class Summary(Flow):  # type: ignore[type-arg]
        @start()
        def summarize(self) -> str:
            return FakeStreamingLLM(answer="From the flow.").call("Summarize")

    class Tool(BaseTool):
        name: str = "lookup"
        description: str = "Runs a flow and asks a model."

        def _run(self, term: str = "x") -> str:
            return f"{Summary().kickoff()} {FakeStreamingLLM(answer='From the tool.').call('Say it')}"

    publisher = RecordingPublisher()
    with publish(publisher, transport=transport):
        make_crew(tool=Tool()).kickoff()  # type: ignore[arg-type]
    events = checked(publisher.data)
    assert list(conversation(events).texts.values()) == ["From the flow.", "From the tool.", ANSWER]
    steps = [event["stepName"] for event in events if event["type"] == "STEP_STARTED"]
    assert steps == ["task:Answer question demo", "flow:summarize"]


def test_a_feedback_classification_in_a_flow_method_is_published_as_the_readme_says() -> None:
    from crewai.flow.human_feedback import human_feedback

    class Provider:
        def request_feedback(self, context: object, flow: object) -> str:
            return "Looks good to me"

    class Review(Flow):  # type: ignore[type-arg]
        @start()
        @human_feedback(
            message="Approve?",
            emit=["approved", "rejected"],
            llm=FakeStreamingLLM(answer="approved"),
            provider=Provider(),
        )
        def draft(self) -> str:
            return "a draft"

    publisher = RecordingPublisher()
    with publish(publisher):
        Review().kickoff()
    # CrewAI's call, inside the application's method: nothing tells it from a call of the application's.
    assert list(conversation(checked(publisher.data)).texts.values()) == ["approved"]
