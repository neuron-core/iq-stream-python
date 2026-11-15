"""The front-end contract: what a real subscriber received, fed to the browser side of @neuron-core/iq-stream in Node
(tests/frontend/check.mjs), reconstructs the answer, the tool call with its arguments and result, and the outcome."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    ANSWER,
    FIRST_TEXT,
    REACT_ANSWER,
    THINKING,
    TOOL_RESULT,
    ScriptedLLM,
    chat_run,
    direct_agent,
    handoff_workflow,
    hitl_agent,
    react_agent,
    repeated_id_agent,
    slow_weather,
    tool_call,
    weather_agent,
)
from iq_stream import IQClient
from iq_stream.testing import TestServers, subscribe_in_background
from live import recording
from llama_index.core.agent.workflow import FunctionAgent, ToolCall
from llama_index.core.base.llms.types import TextBlock
from workflows.events import Event, HumanResponseEvent, InputRequiredEvent
from workflows.handler import WorkflowHandler

from iq_stream_llamaindex import AGUIEncoder, Encoder, publish

CHECK = Path(__file__).resolve().parents[1] / "frontend" / "check.mjs"
JS_DIR = os.environ.get("IQ_STREAM_JS_DIR", "")
BUILT = Path(JS_DIR, "node_modules", "@neuron-core", "iq-stream", "dist", "ag-ui", "index.js")

pytestmark = pytest.mark.skipif(
    not JS_DIR or not BUILT.is_file() or shutil.which("node") is None,
    reason="set IQ_STREAM_JS_DIR to a built checkout of iq-stream-js (with node_modules) and put node on PATH",
)


def check(record: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    done = subprocess.run(
        ["node", str(CHECK), str(path)],
        env={**os.environ, "IQ_STREAM_JS_DIR": JS_DIR},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


async def published(
    client: IQClient,
    url: str,
    handler: WorkflowHandler,
    encoder: Encoder | None = None,
    *,
    answer: bool = False,
    cancel: bool = False,
) -> dict[str, Any]:
    """Publishes the run and returns what a real subscriber received, as a recording."""
    ticket = client.ticket()
    subscription = subscribe_in_background(url, ticket.stream_id, ticket.token)

    def on_event(event: Event) -> None:
        if answer and isinstance(event, InputRequiredEvent):
            handler.ctx.send_event(HumanResponseEvent(response="yes"))
        if cancel and isinstance(event, ToolCall):
            asyncio.get_running_loop().create_task(handler.cancel_run())

    await publish(handler, client.apublisher(ticket.stream_id), encoder=encoder, on_event=on_event)
    return recording(await subscription, "native" if encoder is None else "agui")


def agui() -> AGUIEncoder:
    return AGUIEncoder(thread_id="thread-1")


async def test_native_frames_rebuild_the_answer_the_reasoning_and_the_tool_call(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    record = await published(iq_client, iq_servers.default.url, weather_agent().run(user_msg="Weather in Paris?"))
    seen = check(record, tmp_path)
    assert seen["events"] == len(record["frames"]) and seen["ids"] == [f["id"] for f in record["frames"]]
    assert seen["text"] == FIRST_TEXT + ANSWER and seen["thinking"] == THINKING
    assert seen["tools"] == [{"name": "get_weather", "args": {"city": "Paris"}, "result": TOOL_RESULT}]
    assert seen["last"] == "StopEvent" and seen["lastData"]["result"] == ANSWER
    assert seen["outcome"] == {"type": "end"}


async def test_native_frames_of_a_failed_run_end_with_the_masked_failure_and_fail(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = weather_agent(fail_on_second_call=True).run(user_msg="Weather?")
    seen = check(await published(iq_client, iq_servers.default.url, handler), tmp_path)
    assert seen["last"] == "WorkflowFailedEvent" and seen["lastData"]["message"] == "An error occurred."
    assert seen["outcome"] == {"type": "fail", "reason": "publisher"}


async def test_ag_ui_frames_rebuild_the_messages_in_stream_agent(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = weather_agent().run(user_msg="Weather in Paris?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui()), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"] == FIRST_TEXT + ANSWER
    (call,) = seen["toolCalls"]
    assert call == {"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}
    assert seen["toolResults"] == [{"toolCallId": "call_1", "content": TOOL_RESULT}]
    assert seen["last"]["type"] == "RUN_FINISHED"
    # the tool call joins the assistant message that announced it
    assert seen["roles"].count("assistant") == 2 and "tool" in seen["roles"]


async def test_ag_ui_frames_of_a_failed_run_end_in_run_error_without_a_second_error(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = weather_agent(fail_on_second_call=True).run(user_msg="Weather?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui()), tmp_path)
    assert seen["last"]["type"] == "RUN_ERROR" and seen["last"]["message"] == "An error occurred."
    assert seen["rejected"] is None or "An error occurred." in seen["rejected"]
    assert not any("stream" in failure.lower() for failure in seen["failures"])
    assert seen["text"].startswith(FIRST_TEXT)


async def test_ag_ui_frames_of_a_cancelled_run_finish_with_the_cancelled_outcome(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui(), cancel=True), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["last"]["type"] == "RUN_FINISHED" and seen["last"]["outcome"] == {"type": "cancelled"}


async def test_ag_ui_steps_of_an_agent_workflow_are_accepted_by_the_client(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = handoff_workflow().run(user_msg="Weather in Rome?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui()), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert [call["name"] for call in seen["toolCalls"]] == ["handoff", "get_weather"]
    assert seen["text"] == "Sunny in Rome." and seen["types"].count("STEP_STARTED") == 2


async def test_ag_ui_frames_of_a_human_input_run_carry_the_question_and_the_rest_of_the_answer(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = hitl_agent().run(user_msg="Do it")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui(), answer=True), tmp_path)
    assert seen["rejected"] is None and "CUSTOM" in seen["types"]
    assert seen["toolResults"][0]["content"] == "yes" and seen["text"] == "Done."


async def test_ag_ui_chat_workflow_events_passed_through_rebuild_the_answer(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    encoder = AGUIEncoder(thread_id="thread-1", run_id="run-1")
    seen = check(await published(iq_client, iq_servers.default.url, chat_run(), encoder), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert ANSWER in seen["text"]
    assert any(call["name"] == "get_weather" and call["args"] == {"city": "Paris"} for call in seen["toolCalls"])


async def test_ag_ui_chat_workflow_events_with_a_split_surrogate_pair_rebuild_the_whole_answer(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = chat_run(answer_tokens=["Sunny \ud83d", "\ude00 in ", "Paris."])
    encoder = AGUIEncoder(thread_id="thread-1", run_id="run-1")
    seen = check(await published(iq_client, iq_servers.default.url, handler, encoder), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"].endswith("Sunny \ufffd\ufffd in Paris.")


async def test_ag_ui_frames_of_a_tool_that_changes_its_arguments_in_place_rebuild_one_call_with_its_result(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    announced = asyncio.Event()

    async def sort_numbers(numbers: list[int]) -> str:
        """Sort the numbers."""
        await announced.wait()
        numbers.sort()
        return f"sorted: {numbers}"

    llm = ScriptedLLM(script=[[tool_call("sort_numbers", "c1", numbers=[10, 3, 7])], [TextBlock(text="Done.")]])
    handler = FunctionAgent(name="a", tools=[sort_numbers], llm=llm).run(user_msg="Sort 10 3 7")
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            announced.set()

    await publish(handler, iq_client.apublisher(ticket.stream_id), encoder=agui(), on_event=on_event)
    seen = check(recording(await subscription, "agui"), tmp_path)
    assert seen["toolCalls"] == [{"id": "c1", "name": "sort_numbers", "args": {"numbers": [10, 3, 7]}}]
    assert seen["toolResults"] == [{"toolCallId": "c1", "content": "sorted: [3, 7, 10]"}]


async def test_split_frames_rebuild_the_whole_answer_and_arguments_in_stream_agent(
    iq_servers: TestServers, tmp_path: Path
) -> None:
    # A client held to 1,024-byte frames, the short server's limit, on the default server: the short one's 8 KiB of
    # stream memory is too little for this run.
    client = IQClient(
        iq_servers.default.url, iq_servers.app_key, iq_servers.app_secret, min_open=0, max_frame_bytes=1024
    )
    long = "«sunny» " * 300
    agent = weather_agent(answer_tokens=["Here: ", long], tool_kwargs={"city": "P" * 3000})
    record = await published(client, iq_servers.default.url, agent.run(user_msg="Weather?"), agui())
    assert record["terminal"]["event"] == "iq:end"
    assert all(len(frame["data"].encode()) < 1024 for frame in record["frames"])
    seen = check(record, tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"] == FIRST_TEXT + "Here: " + long
    assert seen["toolCalls"][0]["args"] == {"city": "P" * 3000}
    assert (
        seen["types"].count("TOOL_CALL_ARGS") > 1 and seen["types"].count("TEXT_MESSAGE_CONTENT") >= 5 + 1 + 2
    )  # the long delta split


async def test_ag_ui_frames_of_a_return_direct_tool_rebuild_its_result_as_the_answer(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = direct_agent().run(user_msg="Weather in Paris?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui()), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"] == FIRST_TEXT + TOOL_RESULT and seen["roles"][-1] == "assistant"


async def test_ag_ui_frames_of_calls_that_share_an_id_rebuild_each_call_with_its_own_result(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = repeated_id_agent(parallel=True).run(user_msg="Paris and Rome?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui()), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    cities = {call["id"]: call["args"]["city"] for call in seen["toolCalls"]}
    assert sorted(cities.values()) == ["Paris", "Rome"] and len(cities) == 2
    results = {result["toolCallId"]: result["content"] for result in seen["toolResults"]}
    assert results == {call: f"21 °C and sunny in {city}" for call, city in cities.items()}


async def test_native_frames_of_parallel_calls_that_share_an_id_pair_each_result_with_its_call(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = repeated_id_agent(parallel=True).run(user_msg="Paris and Rome?")
    seen = check(await published(iq_client, iq_servers.default.url, handler), tmp_path)
    assert sorted(tool["args"]["city"] for tool in seen["tools"]) == ["Paris", "Rome"]
    assert all(tool["result"] == f"21 °C and sunny in {tool['args']['city']}" for tool in seen["tools"])


async def test_ag_ui_frames_of_a_react_agent_end_with_the_answer_it_saves(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    seen = check(await published(iq_client, iq_servers.default.url, react_agent().run(user_msg="?"), agui()), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"].endswith('Action Input: {"city": "Paris"}' + REACT_ANSWER)


async def test_ag_ui_frames_with_a_split_surrogate_pair_are_accepted_by_the_client(
    iq_client: IQClient, iq_servers: TestServers, tmp_path: Path
) -> None:
    handler = weather_agent(answer_tokens=["Sunny \ud83d", "\ude00 in ", "Paris."]).run(user_msg="Weather?")
    seen = check(await published(iq_client, iq_servers.default.url, handler, agui()), tmp_path)
    assert seen["rejected"] is None and seen["failures"] == []
    assert seen["text"] == FIRST_TEXT + "Sunny �� in Paris."
