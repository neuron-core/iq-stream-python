"""A model message left open by a generator that failed mid-answer, and the next model call from the same caller:
Haystack's FallbackChatGenerator moves to its next generator with the same callback. The page must show the two
apart, as the saved conversation does, while ordinary messages of every provider shape stay whole."""

from __future__ import annotations

import json
from typing import Any

import pytest
from haystack.dataclasses import ChatMessage, ReasoningContent, StreamingChunk, ToolCall
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import AGUIConverter, AISDKConverter, IQStream
from tests.checks import verify_agui, verify_ai_sdk
from tests.fakes import fallback_agent, tool_turn

QUESTION = [ChatMessage.from_user("Weather in Paris?")]
MAIN = ("main", None)


def published(protocol: str, kind: str) -> tuple[list[str], dict[str, Any]]:
    publisher = RecordingPublisher()
    with IQStream(publisher, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
        result = fallback_agent(kind).run(QUESTION, streaming_callback=stream.callback)
    assert publisher.outcome == "end" and not stream.failed
    return publisher.data, result


def test_ag_ui_a_retried_answer_is_a_message_of_its_own() -> None:
    data, result = published("ag-ui", "text")
    assert result["last_message"].text == "It is sunny in Paris."
    assert verify_agui(data).texts == ["The weather in Par", "It is sunny in Paris."]


def test_ai_sdk_a_retried_answer_is_a_text_part_of_its_own() -> None:
    data, _ = published("ai-sdk", "text")
    assert verify_ai_sdk(data).texts == ["The weather in Par", "It is sunny in Paris."]


def test_ag_ui_a_retried_tool_call_keeps_its_own_arguments_and_the_cut_one_ends() -> None:
    data, result = published("ag-ui", "tool")
    assert result["last_message"].text == "It is sunny in Paris."
    conversation = verify_agui(data)
    assert [(call_id, args) for call_id, _, args, _ in conversation.calls] == [
        ("call_A", '{"city": "Ro'),
        ("call_B", '{"city": "Paris"}'),
    ]
    assert conversation.results == {"call_B": "Sunny in Paris"}


def test_ai_sdk_a_cut_tool_call_is_an_input_error_and_the_retried_one_runs() -> None:
    data, _ = published("ai-sdk", "tool")
    message = verify_ai_sdk(data)
    assert message.tools["call_A"]["state"] == "output-error"
    assert message.tools["call_A"]["error"] == "The tool call's arguments are not valid JSON."
    assert message.tools["call_B"]["input"] == {"city": "Paris"}
    assert message.tools["call_B"]["output"] == "Sunny in Paris"
    assert message.texts == ["It is sunny in Paris."]


# -- the shapes of ordinary messages, which must stay whole ------------------------------------------------------------


def frames(converter: AGUIConverter | AISDKConverter, chunks: list[StreamingChunk]) -> list[str]:
    out = [*converter.start()]
    for chunk in chunks:
        out += converter.convert(chunk, MAIN)
    return [frame.data for frame in [*out, *converter.finish()]]


def agui() -> AGUIConverter:
    return AGUIConverter(thread_id="thread-1", run_id="run-1")


def test_openai_chat_completions_number_text_and_tool_calls_both_from_zero_in_one_message() -> None:
    chunks = [
        StreamingChunk(content="Let me check.", index=0, start=True),
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "get_weather", None, "c1")]),
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, None, '{"city": "Paris"}')]),
        StreamingChunk(content="", index=1, start=True, tool_calls=[ToolCallDelta(1, "get_weather", None, "c2")]),
        StreamingChunk(content="", index=1, tool_calls=[ToolCallDelta(1, None, '{"city": "Rome"}')]),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    data = frames(agui(), chunks)
    conversation = verify_agui(data)
    assert conversation.texts == ["Let me check."]
    text_id = json.loads(data[1])["messageId"]
    assert [parent for *_, parent in conversation.calls] == [text_id, text_id]  # one model message
    assert [args for _, _, args, _ in conversation.calls] == ['{"city": "Paris"}', '{"city": "Rome"}']


def test_a_provider_that_repeats_the_call_s_id_and_name_on_every_fragment_streams_one_call() -> None:
    fragments = ['{"city"', ': "Paris"}']
    chunks = [
        StreamingChunk(content="", index=0, start=number == 0, tool_calls=[ToolCallDelta(0, "get_weather", part, "c1")])
        for number, part in enumerate(fragments)
    ]
    conversation = verify_agui(frames(agui(), [*chunks, StreamingChunk(content="", finish_reason="tool_calls")]))
    assert [(call_id, args) for call_id, _, args, _ in conversation.calls] == [("c1", '{"city": "Paris"}')]


def test_anthropic_shaped_retry_after_thinking_and_text_starts_a_new_message() -> None:
    # A block opens with an empty chunk (start=True), its deltas follow; the retry numbers its blocks from 0 again.
    cut = [
        StreamingChunk(content="", index=0, start=True, reasoning=ReasoningContent("")),
        StreamingChunk(content="", index=0, reasoning=ReasoningContent("The user asks.")),
        StreamingChunk(content="", index=1, start=True),
        StreamingChunk(content="Paris is", index=1),
    ]
    retry = [
        StreamingChunk(content="", index=0, start=True),
        StreamingChunk(content="It is sunny.", index=0),
        StreamingChunk(content="", finish_reason="stop"),
    ]
    conversation = verify_agui(frames(agui(), cut + retry))
    assert conversation.reasoning == ["The user asks."]
    assert conversation.texts == ["Paris is", "It is sunny."]


def test_id_less_calls_retried_at_the_same_index_stay_apart() -> None:
    # MockChatGenerator streams each call whole, without an id unless one is set.
    first = tool_turn([ToolCall(tool_name="get_weather", arguments={"city": "Rome"})], shape="mock", finish=False)
    second = tool_turn([ToolCall(tool_name="get_weather", arguments={"city": "Paris"})], shape="mock")
    tools = verify_ai_sdk(frames(AISDKConverter(message_id="run-1"), first.chunks + second.chunks)).tools
    assert sorted(tool["input"]["city"] for tool in tools.values()) == ["Paris", "Rome"]


def test_ollama_parallel_calls_without_text_stay_one_message() -> None:
    # ollama-haystack numbers its chunks by position and its tool calls from 1, and marks every chunk as a start: the
    # second call's chunk index (1) is the first call's delta index, which must not read as a block used again.
    chunks = [
        StreamingChunk(
            content="", index=0, start=True, tool_calls=[ToolCallDelta(1, "get_weather", '{"city": "Paris"}')]
        ),
        StreamingChunk(
            content="", index=1, start=True, tool_calls=[ToolCallDelta(2, "get_weather", '{"city": "Rome"}')]
        ),
        StreamingChunk(content="", index=2, finish_reason="stop"),
    ]
    data = frames(agui(), chunks)
    conversation = verify_agui(data)
    assert [args for _, _, args, _ in conversation.calls] == ['{"city": "Paris"}', '{"city": "Rome"}']
    assert len({parent for *_, parent in conversation.calls}) == 1


def test_ollama_text_after_parallel_calls_stays_in_their_message() -> None:
    chunks = [
        StreamingChunk(
            content="", index=0, start=True, tool_calls=[ToolCallDelta(1, "get_weather", '{"city": "Paris"}')]
        ),
        StreamingChunk(
            content="", index=1, start=True, tool_calls=[ToolCallDelta(2, "get_weather", '{"city": "Rome"}')]
        ),
        StreamingChunk(content="Checking both.", index=2, start=True),
        StreamingChunk(content="", index=3, finish_reason="stop"),
    ]
    data = frames(agui(), chunks)
    events = [json.loads(item) for item in data]
    assert sum(event["type"] == "TOOL_CALL_END" for event in events) == 2
    # The calls end only when the message does, after its text: one model message.
    ends = [number for number, event in enumerate(events) if event["type"] == "TOOL_CALL_END"]
    text = next(number for number, event in enumerate(events) if event["type"] == "TEXT_MESSAGE_START")
    assert min(ends) > text
    assert verify_agui(data).texts == ["Checking both."]


def test_a_call_at_a_used_index_with_another_provider_id_starts_another_message_without_a_start_flag() -> None:
    chunks = [
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, "get_weather", '{"city": "Ro', "call_A")]),
        StreamingChunk(
            content="", index=0, tool_calls=[ToolCallDelta(0, "get_weather", '{"city": "Paris"}', "call_B")]
        ),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    conversation = verify_agui(frames(agui(), chunks))
    assert [(call_id, args) for call_id, _, args, _ in conversation.calls] == [
        ("call_A", '{"city": "Ro'),
        ("call_B", '{"city": "Paris"}'),
    ]


def test_an_id_less_call_named_again_after_its_arguments_starts_another_message_without_a_start_flag() -> None:
    chunks = [
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, "get_weather", '{"city": "Ro')]),
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, "get_weather", '{"city": "Paris"}')]),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    conversation = verify_agui(frames(agui(), chunks))
    assert [args for _, _, args, _ in conversation.calls] == ['{"city": "Ro', '{"city": "Paris"}']
    assert len({call_id for call_id, *_ in conversation.calls}) == 2


def test_an_id_less_call_cut_off_before_its_arguments_and_started_again_is_two_calls() -> None:
    chunks = [
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "get_weather")]),
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "get_weather")]),
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, None, '{"city": "Paris"}')]),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    conversation = verify_agui(frames(agui(), chunks))
    assert [args for _, _, args, _ in conversation.calls] == ["", '{"city": "Paris"}']


def test_text_that_starts_at_the_index_of_a_cut_off_call_starts_another_message() -> None:
    # Anthropic-shaped: the first answer opened a tool_use block at 0; the retry answers in text at block 0.
    chunks = [
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "get_weather", None, "call_A")]),
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, None, '{"city": "Ro')]),
        StreamingChunk(content="", index=0, start=True),
        StreamingChunk(content="It is sunny.", index=0),
        StreamingChunk(content="", finish_reason="stop"),
    ]
    data = frames(agui(), chunks)
    events = [json.loads(item) for item in data]
    end = next(n for n, e in enumerate(events) if e["type"] == "TOOL_CALL_END")
    text = next(n for n, e in enumerate(events) if e["type"] == "TEXT_MESSAGE_START")
    assert end < text  # the cut-off message ended before the retried answer began
    assert verify_agui(data).texts == ["It is sunny."]


def test_ai_sdk_a_cut_off_call_too_large_for_a_frame_is_an_input_error_without_its_input() -> None:
    cut = '{"text": "' + "n" * 2_000
    chunks = [
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "save_note", None, "call_A")]),
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, None, cut)]),
        StreamingChunk(content="", finish_reason="length"),
    ]
    data = frames(AISDKConverter(message_id="run-1", max_frame_bytes=1024), chunks)
    tool = verify_ai_sdk(data).tools["call_A"]
    assert tool["state"] == "output-error" and tool["input"] == ""
    assert "".join(tool["deltas"]) == cut  # the deltas carried it


def test_ai_sdk_tool_input_with_a_nan_is_written_as_null() -> None:
    chunks = [
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "rate", '{"score": NaN}', "c1")]),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    data = frames(AISDKConverter(message_id="run-1"), chunks)
    available = next(item for item in data if '"tool-input-available"' in item)
    assert json.loads(available, parse_constant=lambda name: pytest.fail(f"{name} is not JSON"))["input"] == {
        "score": None
    }


async def published_async(protocol: str) -> tuple[list[str], dict[str, Any]]:
    publisher = RecordingPublisher()
    async with IQStream(publisher, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
        result = await fallback_agent("tool", retry_in_a_thread=True).run_async(
            QUESTION, streaming_callback=stream.callback
        )
    assert publisher.outcome == "end" and not stream.failed
    return publisher.data, result


async def test_ag_ui_a_cut_off_call_ends_before_a_retry_from_another_thread_gets_its_result() -> None:
    data, result = await published_async("ag-ui")
    assert result["last_message"].text == "It is sunny in Paris."
    events = [json.loads(item) for item in data]
    end_a = next(n for n, e in enumerate(events) if e["type"] == "TOOL_CALL_END" and e["toolCallId"] == "call_A")
    result_b = next(n for n, e in enumerate(events) if e["type"] == "TOOL_CALL_RESULT")
    assert end_a < result_b
    conversation = verify_agui(data)
    assert conversation.results == {"call_B": "Sunny in Paris"}
    assert conversation.texts == ["It is sunny in Paris."]


async def test_ai_sdk_a_retry_from_another_thread_keeps_one_step_per_model_call() -> None:
    data, _ = await published_async("ai-sdk")
    message = verify_ai_sdk(data)
    types = [json.loads(item)["type"] for item in data]
    assert types.count("start-step") == types.count("finish-step") == message.steps == 2
    assert message.tools["call_A"]["state"] == "output-error"
    assert message.tools["call_B"]["output"] == "Sunny in Paris"
    assert message.texts == ["It is sunny in Paris."]
    error_a = next(n for n, item in enumerate(data) if '"tool-input-error"' in item)
    output_b = next(n for n, item in enumerate(data) if '"tool-output-available"' in item)
    assert error_a < output_b
