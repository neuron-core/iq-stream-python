"""NativeEncoder: the event class as SSE event name and a safe JSON projection as data; nothing private leaves."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from ag_ui.core import TextMessageContentEvent
from fakes import (
    BIG_RESULT,
    SYSTEM_PROMPT,
    ScriptedLLM,
    big_weather,
    chat_run,
    get_weather,
    needs_ag_ui_workflow,
    opaque_weather,
    repeated_id_agent,
    tool_call,
    weather_agent,
)
from iq_stream.testing import AsyncRecordingPublisher
from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.agent.workflow.workflow_events import AgentStreamStructuredOutput
from llama_index.core.base.llms.types import TextBlock
from workflows.events import Event, InputRequiredEvent, StopEvent

from iq_stream_llamaindex import AGUIEncoder, Encoder, NativeEncoder, publish
from iq_stream_llamaindex.encoders.base import DEFAULT_MAX_FIELD_BYTES


async def frames_of(handler: object, encoder: Encoder | None = None) -> list[tuple[str | None, str]]:
    publisher = AsyncRecordingPublisher()
    await publish(handler, publisher, encoder=encoder)  # type: ignore[arg-type]
    return [(frame.event, frame.data) for frame in publisher.frames]


def payloads(frames: list[tuple[str | None, str]], name: str) -> list[dict[str, object]]:
    return [json.loads(data) for event, data in frames if event == name]


async def test_agent_stream_frames_carry_the_delta_not_the_cumulative_response_nor_raw() -> None:
    frames = await frames_of(weather_agent().run(user_msg="Weather?"))
    streamed = payloads(frames, "AgentStream")
    assert streamed and all(set(frame) <= {"delta", "agent", "thinking_delta"} for frame in streamed)
    assert streamed[0] == {
        "delta": "",
        "agent": "weather_agent",
        "thinking_delta": "The user wants weather; call the tool.",
    }
    # one frame per delta, a few bytes each, whatever the length of the answer
    assert max(len(data) for event, data in frames if event == "AgentStream" and "thinking" not in data) < 80


@pytest.mark.parametrize("encoder", [NativeEncoder(), AGUIEncoder(thread_id="t")], ids=["native", "agui"])
async def test_the_system_prompt_and_the_chat_history_never_reach_a_frame(encoder: Encoder) -> None:
    frames = await frames_of(weather_agent().run(user_msg="PRIVATE-USER-HISTORY-MARKER"), encoder)
    assert not any("SECRET-SYSTEM-PROMPT" in data or "PRIVATE-USER-HISTORY" in data for _, data in frames)
    assert SYSTEM_PROMPT.startswith("SECRET-SYSTEM-PROMPT")
    assert "AgentInput" not in {event for event, _ in frames}


async def test_a_200_kib_tool_result_is_cut_to_max_field_bytes_with_the_marker() -> None:
    frames = await frames_of(weather_agent(tool=big_weather).run(user_msg="Weather?"))
    (result,) = payloads(frames, "ToolCallResult")
    assert result["truncated"] is True
    content = str(result["content"])
    assert content.endswith("…") and len(content.encode()) <= DEFAULT_MAX_FIELD_BYTES + len("…".encode())
    assert BIG_RESULT.startswith(content[:-1])
    assert all(len(data.encode()) < 65_536 for _, data in frames)


async def test_a_tool_result_json_has_no_type_for_does_not_lose_its_frame() -> None:
    frames = await frames_of(weather_agent(tool=opaque_weather).run(user_msg="Weather?"))
    (result,) = payloads(frames, "ToolCallResult")
    assert result["content"] == "an opaque report" and result["is_error"] is False


def test_long_tool_arguments_stay_json_and_are_marked() -> None:
    encoder = NativeEncoder(max_field_bytes=64)
    (frame,) = encoder.encode(ToolCall(tool_name="search", tool_kwargs={"query": "q" * 500, "limit": 5}, tool_id="c"))
    payload = json.loads(frame[1])
    assert payload["truncated"] is True and payload["tool_kwargs"]["limit"] == 5
    assert len(json.dumps(payload["tool_kwargs"], separators=(",", ":"), ensure_ascii=False).encode()) <= 64


async def test_an_input_request_is_published_with_its_undeclared_fields_flattened() -> None:
    class Approval(InputRequiredEvent):
        tool: str

    encoder = NativeEncoder()
    (frame,) = encoder.encode(Approval(tool="book_trip", prefix="Book it?", amount=120))
    assert frame == ("Approval", '{"tool":"book_trip","prefix":"Book it?","amount":120}')


def test_an_event_class_name_the_server_refuses_becomes_event() -> None:
    accented = type("Événement", (Event,), {})
    long = type("L" * 65, (Event,), {})
    encoder = NativeEncoder()
    assert [name for name, _ in encoder.encode(accented())] == ["Event"]
    assert [name for name, _ in encoder.encode(long())] == ["Event"]
    assert [name for name, _ in encoder.encode(type("Good.Name-1", (Event,), {})())] == ["Good.Name-1"]


def test_include_and_exclude_choose_events_but_the_terminal_event_always_goes_out() -> None:
    stream = AgentStream(delta="Hi", response="Hi", current_agent_name="a")
    call = ToolCall(tool_name="t", tool_kwargs={}, tool_id="1")
    only_tools = NativeEncoder(include=[ToolCall])
    assert list(only_tools.encode(stream)) == []
    assert [name for name, _ in only_tools.encode(call)] == ["ToolCall"]
    assert [name for name, _ in only_tools.encode(StopEvent(result="ok"))] == ["StopEvent"]
    no_text = NativeEncoder(exclude=[AgentStream])
    assert list(no_text.encode(stream)) == []
    assert [name for name, _ in no_text.encode(call)] == ["ToolCall"]


def test_a_chunk_with_only_tool_call_deltas_is_not_published() -> None:
    encoder = NativeEncoder()
    assert list(encoder.encode(AgentStream(delta="", response="", current_agent_name="a"))) == []


def test_a_replayed_tool_call_is_announced_once_and_its_result_always() -> None:
    encoder = NativeEncoder()
    call = ToolCall(tool_name="ask", tool_kwargs={"q": 1}, tool_id="q1")
    assert len(list(encoder.encode(call))) == 1
    assert list(encoder.encode(call)) == []


def test_results_of_every_kind_are_json() -> None:
    encoder = NativeEncoder()

    class Done(StopEvent):
        pages: int

    assert list(encoder.encode(StopEvent(result={"a": [1, 2]}))) == [("StopEvent", '{"result":{"a":[1,2]}}')]
    assert list(encoder.encode(StopEvent(result=3))) == [("StopEvent", '{"result":3}')]
    assert list(encoder.encode(StopEvent())) == [("StopEvent", '{"result":null}')]
    assert list(encoder.encode(Done(pages=2))) == [("Done", '{"pages":2}')]
    long = list(NativeEncoder(max_field_bytes=10).encode(StopEvent(result="y" * 50)))
    assert json.loads(long[0][1]) == {"result": "y" * 10 + "…", "truncated": True}


def test_a_structured_output_is_published_as_json() -> None:
    (frame,) = NativeEncoder().encode(AgentStreamStructuredOutput(output={"city": "Paris"}))
    assert frame == ("AgentStreamStructuredOutput", '{"output":{"city":"Paris"}}')


@needs_ag_ui_workflow
def test_an_ag_ui_event_keeps_its_wire_json_under_its_class_name() -> None:
    from llama_index.protocols.ag_ui.events import TextMessageContentWorkflowEvent

    event = TextMessageContentWorkflowEvent(message_id="m1", delta="Hi")
    (frame,) = NativeEncoder().encode(event)
    assert frame == ("TextMessageContentWorkflowEvent", '{"type":"TEXT_MESSAGE_CONTENT","messageId":"m1","delta":"Hi"}')
    assert isinstance(event, TextMessageContentEvent)


def test_abort_frames_say_cancelled_or_masked_error() -> None:
    encoder = NativeEncoder()
    assert list(encoder.abort(asyncio.CancelledError())) == [("RunAborted", '{"reason":"cancelled"}')]
    assert list(encoder.abort(OSError("disk full at /secret"))) == [
        ("RunAborted", '{"reason":"error","message":"An error occurred."}')
    ]
    exposed = NativeEncoder(expose_errors=True)
    assert json.loads(next(iter(exposed.abort(OSError("disk full"))))[1]) == {
        "reason": "error",
        "message": "disk full",
        "type": "OSError",
    }


async def test_a_failed_runs_error_is_exposed_with_its_type_only_when_asked() -> None:
    frames = await frames_of(
        weather_agent(fail_on_second_call=True).run(user_msg="Weather?"), NativeEncoder(expose_errors=True)
    )
    (failed,) = payloads(frames, "WorkflowFailedEvent")
    assert failed["type"] == "RuntimeError" and "exploded" in str(failed["message"])


async def test_an_agent_that_does_not_stream_publishes_its_answer_in_agent_output() -> None:
    llm = ScriptedLLM(script=[[tool_call("get_weather", "c1", city="Oslo")], [TextBlock(text="Sunny in Oslo.")]])
    agent = FunctionAgent(name="plain", tools=[get_weather], llm=llm, streaming=False)
    frames = await frames_of(agent.run(user_msg="Oslo?"))
    assert "AgentStream" not in {event for event, _ in frames}
    outputs = payloads(frames, "AgentOutput")
    assert outputs[0]["tool_calls"] == [{"tool_id": "c1", "tool_name": "get_weather", "tool_kwargs": {"city": "Oslo"}}]
    assert outputs[-1]["content"] == "Sunny in Oslo."
    assert payloads(frames, "StopEvent") == [{"result": "Sunny in Oslo."}]


def test_a_tool_result_carries_its_flags() -> None:
    from llama_index.core.tools import ToolOutput

    output = ToolOutput(tool_name="t", content="ok", raw_input={}, raw_output=object())
    (frame,) = NativeEncoder().encode(
        ToolCallResult(tool_name="t", tool_kwargs={}, tool_id="1", tool_output=output, return_direct=True)
    )
    assert json.loads(frame[1]) == {
        "tool_id": "1",
        "call_id": "1",
        "tool_name": "t",
        "content": "ok",
        "is_error": False,
        "return_direct": True,
    }


def test_max_field_bytes_is_positive() -> None:
    with pytest.raises(ValueError, match="max_field_bytes"):
        NativeEncoder(max_field_bytes=0)


@pytest.mark.parametrize("parallel", [False, True], ids=["two-steps", "one-step"])
async def test_tool_calls_that_share_an_id_are_all_published(parallel: bool) -> None:
    frames = await frames_of(repeated_id_agent(parallel=parallel).run(user_msg="Paris and Rome?"))
    calls = payloads(frames, "ToolCall")
    assert sorted(call["tool_kwargs"]["city"] for call in calls) == ["Paris", "Rome"]  # type: ignore[index]
    assert {call["tool_id"] for call in calls} == {"get_weather"}  # LlamaIndex's id, as the provider gave it
    contents = sorted(str(result["content"]) for result in payloads(frames, "ToolCallResult"))
    assert contents == ["21 °C and sunny in Paris", "21 °C and sunny in Rome"]
    if parallel:  # both calls were announced before either result
        assert [event for event, _ in frames].index("ToolCallResult") > [event for event, _ in frames].index(
            "ToolCall"
        ) + 1


def test_a_call_like_an_earlier_one_that_has_its_result_is_a_new_call() -> None:
    from llama_index.core.tools import ToolOutput

    encoder = NativeEncoder()
    call = ToolCall(tool_name="get_weather", tool_kwargs={"city": "Paris"}, tool_id="get_weather")
    output = ToolOutput(tool_name="get_weather", content="sunny", raw_input={}, raw_output="sunny")
    result = ToolCallResult(**call.model_dump(), tool_output=output, return_direct=False)
    published = [name for event in (call, call, result, call) for name, _ in encoder.encode(event)]
    assert published == ["ToolCall", "ToolCallResult", "ToolCall"]  # the replay is left out, the new call is not


async def test_one_encoder_serves_runs_one_after_another() -> None:
    encoder = NativeEncoder()
    for _ in (1, 2):
        handler = weather_agent().run(user_msg="Weather?")
        frames = await frames_of(handler, encoder)
        assert json.loads(frames[0][1]) == {"run_id": handler.run_id}
        assert [event for event, _ in frames].count("ToolCall") == 1
    call = ToolCall(tool_name="get_weather", tool_kwargs={"city": "Paris"}, tool_id="call_1")
    assert len(list(encoder.encode(call))) == 1  # a run stopped before this call had its result
    list(encoder.start("run-3"))
    assert len(list(encoder.encode(call))) == 1


async def test_lone_surrogates_become_replacement_characters_and_no_frame_is_lost() -> None:
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "21 \udcc3C in Paris"  # decoded with surrogateescape

    agent = weather_agent(tool=get_weather, answer_tokens=["Sunny \ud83d", "\ude00."])
    frames = await frames_of(agent.run(user_msg="Weather?"))
    assert [event for event, _ in frames][-3:] == ["AgentStream", "AgentOutput", "StopEvent"]
    assert payloads(frames, "ToolCallResult")[0]["content"] == "21 �C in Paris"
    assert payloads(frames, "AgentOutput")[-1]["content"] == "Sunny ��."
    assert payloads(frames, "StopEvent") == [{"result": "Sunny ��."}]


def test_non_finite_numbers_are_null_so_that_every_frame_is_json() -> None:
    def strict(data: str) -> object:
        return json.loads(data, parse_constant=lambda constant: pytest.fail(f"{constant} is not JSON"))

    encoder = NativeEncoder()
    (stop,) = encoder.encode(StopEvent(result={"mean": float("nan"), "max": float("inf"), "n": 0}))
    assert strict(stop[1]) == {"result": {"mean": None, "max": None, "n": 0}}
    (bare,) = encoder.encode(StopEvent(result=float("nan")))
    assert strict(bare[1]) == {"result": None}
    (structured,) = encoder.encode(AgentStreamStructuredOutput(output={"confidence": float("nan")}))
    assert strict(structured[1]) == {"output": {"confidence": None}}
    (call,) = encoder.encode(ToolCall(tool_name="t", tool_kwargs={"x": float("-inf")}, tool_id="c"))
    assert strict(call[1]) == {"tool_id": "c", "call_id": "c", "tool_name": "t", "tool_kwargs": {"x": None}}
    (custom,) = encoder.encode(type("Stats", (Event,), {})(mean=float("nan")))
    assert strict(custom[1]) == {"mean": None}
    assert encoder.encode(StopEvent(result={"label": "NaN"})) == [("StopEvent", '{"result":{"label":"NaN"}}')]


# ---------------------------------------------------------------------------------------------------- round two


def result_of(call: ToolCall, content: str) -> ToolCallResult:
    from llama_index.core.tools import ToolOutput

    output = ToolOutput(tool_name=call.tool_name, content=content, raw_input={}, raw_output=content)
    return ToolCallResult(
        tool_name=call.tool_name,
        tool_kwargs=call.tool_kwargs,
        tool_id=call.tool_id,
        tool_output=output,
        return_direct=False,
    )


def test_a_result_whose_arguments_the_tool_changed_in_place_closes_its_call() -> None:
    encoder = NativeEncoder()
    numbers = [10, 3, 7]
    call = ToolCall(tool_name="sort_numbers", tool_kwargs={"numbers": numbers}, tool_id="get_weather")
    (announced,) = encoder.encode(call)
    numbers.sort()  # the tool sorts the list it was given, which the result's arguments are
    (result,) = encoder.encode(result_of(call, "sorted"))
    assert json.loads(announced[1])["call_id"] == json.loads(result[1])["call_id"] == "get_weather"
    # the call is closed: one with the same id and the model's arguments later (Ollama's ids repeat) is a new call
    again = ToolCall(tool_name="sort_numbers", tool_kwargs={"numbers": [10, 3, 7]}, tool_id="get_weather")
    assert [json.loads(data)["call_id"] for _, data in encoder.encode(again)] == ["get_weather#2"]


def test_arguments_whose_keys_cannot_be_sorted_still_publish_the_call_and_its_result() -> None:
    encoder = NativeEncoder()
    call = ToolCall(tool_name="score", tool_kwargs={"table": {1: "one", "two": 2}}, tool_id="c9")
    frames = [*encoder.encode(call), *encoder.encode(result_of(call, "ok"))]
    assert [name for name, _ in frames] == ["ToolCall", "ToolCallResult"]
    assert json.loads(frames[0][1])["tool_kwargs"] == {"table": {"1": "one", "two": 2}}


async def test_parallel_calls_that_share_an_id_pair_with_their_results_by_call_id() -> None:
    frames = await frames_of(repeated_id_agent(parallel=True).run(user_msg="Paris and Rome?"))
    calls = {str(call["call_id"]): call["tool_kwargs"]["city"] for call in payloads(frames, "ToolCall")}  # type: ignore[index]
    assert sorted(calls) == ["get_weather", "get_weather#2"]
    results = {str(result["call_id"]): result["content"] for result in payloads(frames, "ToolCallResult")}
    assert results == {call_id: f"21 °C and sunny in {city}" for call_id, city in calls.items()}


def test_an_ag_ui_event_passed_through_with_a_lone_surrogate_is_published_with_a_replacement_character() -> None:
    from ag_ui.core import TextMessageChunkEvent

    event = TextMessageChunkEvent(message_id="m1", delta="Sunny \ud83d")
    (frame,) = NativeEncoder().encode(event)  # type: ignore[arg-type]
    assert frame == ("TextMessageChunkEvent", '{"type":"TEXT_MESSAGE_CHUNK","messageId":"m1","delta":"Sunny \ufffd"}')


@needs_ag_ui_workflow
async def test_ag_ui_chat_workflow_events_with_a_split_surrogate_pair_are_all_published() -> None:
    frames = await frames_of(chat_run(answer_tokens=["Sunny \ud83d", "\ude00 in ", "Paris."]))
    chunks = [json.loads(data).get("delta") or "" for name, data in frames if name == "TextMessageChunkWorkflowEvent"]
    assert "".join(chunks).endswith("Sunny \ufffd\ufffd in Paris.")
    snapshots = [data for name, data in frames if name == "MessagesSnapshotWorkflowEvent"]
    assert "Sunny \ufffd\ufffd in Paris." in snapshots[-1]  # the final history


class FailsOnce(NativeEncoder):
    """Fails on its first tool call, after it has counted it, as an encoder may on a value it cannot handle."""

    failed = False

    def _call(self, tool_id: str, call_id: str, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.failed:
            self.failed = True
            raise TypeError("cannot encode this")
        return super()._call(tool_id, call_id, tool_name, kwargs)


def test_a_call_whose_frame_failed_to_encode_is_announced_when_it_comes_again() -> None:
    encoder = FailsOnce()
    call = ToolCall(tool_name="ask", tool_kwargs={"q": 1}, tool_id="q1")
    with pytest.raises(TypeError):
        encoder.encode(call)
    # its replay (a human-input wait) is the first frame of the call a page sees, under the call's own id
    assert [json.loads(data)["call_id"] for _, data in encoder.encode(call)] == ["q1"]
