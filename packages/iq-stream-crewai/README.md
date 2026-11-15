# iq-stream-crewai

Publishes a [CrewAI](https://github.com/crewAIInc/crewAI) run to browsers through an [IQ-Stream server](https://github.com/neuron-core/iq-stream). Everything CrewAI emits inside a `with publish(...)` block (the model's tokens, tool calls and their results, tasks, flow steps) goes to one stream as it happens, as [AG-UI](https://docs.ag-ui.com) events by default or as CrewAI's own events. The block closes the stream with `end` when it exits normally and with `fail` when it raises. The browser reads the stream with `StreamAgent` from [`@neuron-core/iq-stream`](https://www.npmjs.com/package/@neuron-core/iq-stream), or with any AG-UI client.

It is a thin layer over the core client, [`iq-stream`](https://pypi.org/project/iq-stream/), which owns the connection to the server: one request per answer, keepalives while a tool runs, resuming after a lost request, and the close line.

```sh
pip install iq-stream-crewai
```

## Publish a crew

```python
from iq_stream import IQClient
from iq_stream_crewai import publish

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_APP_KEY and IQ_STREAM_APP_SECRET


def answer(stream_id: str, question: str) -> None:
    """Runs where the crew runs: a thread, a task or a queue job."""
    with publish(iq, stream_id):  # the request to the server opens now
        result = crew.kickoff(inputs={"question": question})
        save_answer(stream_id, result.raw)  # before the close line: a late browser loads it from you
    # the block closes the stream with "end", or with "fail" if it raised


ticket = iq.ticket()  # what the browser needs: stream_id, token, url, expires_at
answer(ticket.stream_id, "What is the capital of France?")
```

`crew` stands for your crew and `save_answer()` for your own persistence. The block may hold anything that makes CrewAI emit events: `crew.kickoff()`, `flow.kickoff()`, `agent.kickoff()`, several kickoffs, or a `Crew(stream=True)` whose chunks your code iterates itself, inside the block. CrewAI needs to know nothing about the stream: no id goes into `inputs`.

The block taps CrewAI through a scoped stream sink, the mechanism CrewAI's own frame streaming is built on. It sees every event emitted in the block's context and in the threads CrewAI starts from it, in the thread that emits it, before any listener of the global event bus: several crews in one process never see each other's events, nothing overtakes an event on its way to the stream, and nothing is left to flush when `kickoff()` returns. Each event is copied, encoded under one lock per stream and handed to the core's `send()`, which only queues.

In asyncio, the same object is used with `async with`. The adapter still uses the core's synchronous publisher, because CrewAI emits from worker threads even in `akickoff()`; the closing frames go out before anything is awaited, and the server's answer is awaited in a thread of its own, not in the loop's default executor, where each `kickoff_async()` holds a thread for its whole run.

```python
import asyncio

from iq_stream import IQClient
from iq_stream_crewai import publish

iq = IQClient()


async def answer(stream_id: str, question: str) -> None:
    async with publish(iq, stream_id):  # a cancelled task closes the stream with "fail"
        result = await crew.akickoff(inputs={"question": question})
        await asave_answer(stream_id, result.raw)


asyncio.run(answer(iq.new_stream_id(), "What is the capital of France?"))
```

### Behind a route

CrewAI serves no HTTP of its own; a route starts the run in a worker and answers with the ticket at once. With `StreamAgent` in the browser, the request body is AG-UI's `RunAgentInput`, and the run's `threadId` and `runId` are the client's own:

```python
from concurrent.futures import ThreadPoolExecutor

from ag_ui.core import RunAgentInput
from fastapi import FastAPI

from iq_stream import IQClient
from iq_stream_crewai import publish

app = FastAPI()
iq = IQClient()
workers = ThreadPoolExecutor(max_workers=8)
answers: dict[str, str] = {}  # stands for your database


def answer(stream_id: str, run: RunAgentInput) -> None:
    question = next(message.content for message in reversed(run.messages) if message.role == "user")
    with publish(iq, stream_id, thread_id=run.thread_id, run_id=run.run_id):
        result = build_crew().kickoff(inputs={"question": question})
        answers[stream_id] = result.raw


@app.post("/chat")
def chat(run: RunAgentInput) -> dict[str, str | int]:
    ticket = iq.ticket()
    workers.submit(answer, ticket.stream_id, run)
    return ticket.to_dict()  # streamId, token, url, expiresAt


@app.get("/chat/{stream_id}")
def saved(stream_id: str) -> dict[str, str | None]:
    """What a browser that missed the stream (iq:gone) loads instead."""
    return {"answer": answers.get(stream_id)}
```

The worker must enter the block within 30 seconds of the browser subscribing, the time a stream waits for its publisher; a queue that can be slower calls `iq.reserve(stream_id)` in the route. The core holds the close line back until the stream is three seconds old (`min_open`), so a browser that subscribes just after a quick answer still receives it.

### CrewAI's own events, and tokens

Without streaming on the model, a run publishes its tool calls, tasks and steps, and each answer as one block when the model call returns. `enable_token_streaming()` turns streaming on for the models of crews, agents or models, as `Crew(stream=True)` does inside its kickoff (`LLM(stream=True)` does it for one model). `protocol="crewai"` publishes CrewAI's own events instead of AG-UI's:

```python
from iq_stream import IQClient
from iq_stream_crewai import enable_token_streaming, publish

iq = IQClient()
enable_token_streaming(crew)

with publish(iq, protocol="crewai") as run:  # without a stream id, a new one: run.stream_id
    crew.kickoff(inputs={"question": "What is the capital of France?"})
print(run.stream_id, run.frames_sent, run.result)
```

## What the browser needs

With the default protocol the frames are AG-UI events, unnamed, as AG-UI's own SSE encoding writes them. `StreamAgent` from `@neuron-core/iq-stream/ag-ui` is an `AbstractAgent` of `@ag-ui/client` that posts the run's input with your `start()` and reads the events from the ticket's URL, so CopilotKit or any UI built on `@ag-ui/client` uses it as it would an `HttpAgent`:

```ts
import { StreamAgent } from '@neuron-core/iq-stream/ag-ui';

const agent = new StreamAgent({
  start: async (input) =>
    (await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) })).json(),
});
await agent.runAgent(); // agent.messages: the assistant's text, its tool calls and their results
```

A failed or cancelled run reaches the page twice, as `RUN_ERROR` (or `RUN_FINISHED` with the `cancelled` outcome) and then as `iq:fail` with reason `publisher`; `StreamAgent` ends the run once, on the first. A flow paused for human feedback ends with `RUN_FINISHED` and AG-UI's `interrupt` outcome, which `@ag-ui/client` hands to `onRunFinishedEvent` as `outcome: "interrupt"` with the `interrupts`. With `protocol="crewai"` each frame is named after its CrewAI event type, and the page listens for the names it wants: `subscribe(url, handlers, { events: ['llm_stream_chunk', 'tool_usage_finished', …] })` from `@neuron-core/iq-stream/client`. There is no front-end client for CrewAI's native events; the page keeps its own state.

## What is published

| CrewAI event | `ag-ui` frames | `crewai` frame members |
| --- | --- | --- |
| block entered | `RUN_STARTED` | none |
| `llm_stream_chunk`, text | `TEXT_MESSAGE_START` (once per model call, `messageId` = its `call_id`, `name` = the agent's role), `TEXT_MESSAGE_CONTENT` | `call_id`, `chunk` |
| `llm_stream_chunk`, tool call | `TOOL_CALL_START` (`parentMessageId` = the call), `TOOL_CALL_ARGS` | `call_id`, `chunk`, `tool_call` (`id`, `name`, `index`; never the accumulated arguments) |
| `llm_thinking_chunk` | `REASONING_START`, `REASONING_MESSAGE_START`, `REASONING_MESSAGE_CONTENT`, closed by the call's next event | `call_id`, `chunk` |
| `llm_call_completed`, `llm_call_failed` | ends what the call left open; a call that streamed nothing and returned text: the whole text message | `call_id`, `finish_reason`, `usage` / `error` |
| `tool_usage_started` | binds to the streamed call of that tool and agent with the same arguments (else the oldest); a call no model streamed is announced whole | `tool_name`, `tool_args`, `agent_id`, `agent_role`, `task_id` |
| `tool_usage_finished`, `tool_usage_error` | `TOOL_CALL_RESULT` (`{"error": …}` for an error) | the same, with `output`, `from_cache` / `error` |
| `task_started`, `task_completed`, `task_failed` | `STEP_STARTED`, `STEP_FINISHED` (`task:<name>`) | `task_id`, `task_name`, `output` / `error` |
| `method_execution_started`, `_finished`, `_failed` | `STEP_STARTED`, `STEP_FINISHED` (`flow:<method>`) | `flow_name`, `method_name`, `error` |
| `human_feedback_requested`, `flow_paused` | `CUSTOM` named `crewai.<type>`; a paused flow also makes the run end interrupted | `flow_name`, `method_name`, `message`, `emit`, `request_id` / `flow_id` |
| `agent_execution_*`, `crew_kickoff_*`, `flow_started`, `flow_finished`, `flow_failed`, `human_feedback_received` | none | their ids (`flow_id` included), names, `output`, `error`, `total_tokens` |
| block exits | `RUN_FINISHED`; with outcome `interrupt` while a flow is paused, or `cancelled`; or `RUN_ERROR` | none |

Every `crewai` frame also has `seq`, the event's `emission_sequence`. Members without a value are left out. An event that is already an AG-UI event (as flows written for `ag-ui-crewai` emit them) is forwarded as it is.

What CrewAI does for itself is left out of both protocols:

- the flows it runs inside every agent and for memory;
- its own agents and tasks, with everything they emit: a task's guardrail (the "Guardrail Agent"), an agent's plan (`planning_config`, `reasoning=True`), the planner of `Crew(planning=True)`, memory's saves and searches, knowledge queries;
- the model calls that name no agent which it makes around an agent's run: memory's analysis, the conversion of an output (`output_pydantic`, `output_json`, `response_format`), and the summary of a context window that is full.

Every other model call is published, whoever made it: the agent's own (its final answer forced at `max_iter` included, and those of a model wrapper that does not pass the agent on), and `llm.call()` in the block, in a flow method or in a tool, a flow your tool kicks off included. A model call your task callback makes itself is left out: in CrewAI's events nothing tells it from an output's conversion.

## Which close line

- **The block exits normally**: `RUN_FINISHED`, then `end`. Save the answer inside the block, after `kickoff()`: a browser that arrives after the close receives `iq:gone` and loads it from you. CrewAI has finished its own writes (task callbacks, `output_file`, memory) when `kickoff()` returns.
- **The block raises**: `RUN_ERROR` with `code` `"error"` and the message masked (with `expose_errors=True`, the exception's class and text), then `fail`; the exception goes on unchanged.
- **The block is cancelled** (`asyncio.CancelledError`, `KeyboardInterrupt`): `RUN_FINISHED` with the `cancelled` outcome, so the page can tell a stop from a failure, then `fail`, or `end` with `close_on_cancel="end"`. Only asyncio runs can be cancelled; a cancelled run leaves CrewAI's worker threads running to completion, and what they emit after the block is not published.
- **A flow pauses for human feedback** (`@human_feedback` with an asynchronous provider): `kickoff()` returns `HumanFeedbackPending`, the block exits normally, and the stream ends with `end` after a `CUSTOM` `crewai.flow_paused` frame and `RUN_FINISHED` with the `interrupt` outcome: one interrupt per paused flow, whose `id` is the flow's id (what `Flow.from_pending()` takes), `reason` `"human_feedback"` and `message` the flow's question. The resumed flow is a new run on a new stream.
- **The process dies**: nothing closes the stream; its browsers get `iq:fail` with `publisher_idle` after the server's idle timeout.

Publishing never breaks the run. When the publisher gives up (the server refused, or cannot be reached), the adapter stops encoding and the crew goes on; the reason is in `run.publisher.error`. With the client's `on_error="raise"`, the block raises that error when it exits normally, after your code has run. The adapter's own failures, which never leave the sink, are in `run.errors` and logged once per stream, without event data, on the `iq_stream_crewai` logger.

## Reference

```text
publish(target, stream_id=None, *, protocol="ag-ui", thread_id=None, run_id=None, include=None, exclude=(),
        max_field_bytes=16_384, expose_errors=False, close_on_cancel="fail", transport="auto") -> CrewAIStream
```

`target` is an `IQClient`, whose publisher for `stream_id` (a new id when `None`) is created when the block is entered, or a synchronous publisher (`PublisherLike`) that the block takes over and closes. `publish()` itself sends nothing.

| Option | Default | |
| --- | --- | --- |
| `protocol` | `"ag-ui"` | `"ag-ui"`, `"crewai"`, or an `Encoder` of your own. |
| `thread_id`, `run_id` | the stream id, a new uuid | AG-UI's ids of the run. |
| `include`, `exclude` | `EVENT_TYPES` | The CrewAI event types the encoder sees. Without `llm_stream_chunk`, each answer is published when its call returns. |
| `max_field_bytes` | `16_384` | Tool results, task outputs and other text values are cut to this many bytes of UTF-8, at a character's boundary, with a marker (`… [N bytes truncated]`); `None` cuts nothing. |
| `expose_errors` | `False` | Publish error messages as they are; by default each one is `An error occurred.` |
| `close_on_cancel` | `"fail"` | The close line after a cancellation (`iq_stream.Outcome`). |
| `transport` | `"auto"` | `"sink"` (CrewAI's scoped stream sink), or `"bus"` (handlers on the global event bus). `"auto"` takes the sink. |

`CrewAIStream` has `stream_id`, `thread_id`, `run_id`, `publisher` (the core's, once entered), `frames_sent`, `errors` and `result` (the core's `PublishResult` once the block has exited), and `send(data, event=None)` for a frame of your own, in order with the run's. `enable_token_streaming(*crews_agents_or_models)` turns streaming on for their models. `EVENT_TYPES` lists the event types the adapter knows.

An `Encoder` has four methods: `snapshot(source, event)` runs in the emitting thread and copies what the encoder needs (or returns `None` to skip the event), and `start()`, `frames(snapshot)` and `finish(error)` return `(event name or None, data)` pairs, under the stream's lock.

## Limits

- **Frame size.** A frame must fit the publisher's `max_frame_bytes` (64 KiB, or the app's own limit). Text, reasoning and tool-argument deltas longer than that are split over several frames; a value that cannot be split (a tool result, an error message, a long name) is cut with a marker; a frame that still does not fit is dropped and recorded in `run.errors`. A stream holds at most 10 MiB and lasts at most 30 minutes by default: about 90,000 tokens of answer, since an AG-UI text frame costs about 100 bytes per token.
- **Tool results are matched by tool, agent and arguments.** CrewAI's tool events carry no tool call id; two concurrent calls of the same tool by the same agent with the same arguments get their results in the order they started.
- **Models without function calling** stream CrewAI's ReAct text (`Thought:`, `Action:`, `Final Answer:`), published as the model wrote it; the tool call is announced when the tool starts.
- **Every model call is a message.** A crew with several tasks or agents publishes one assistant message per model call of its agents, intermediate ones included; the `name` of each message is the agent's role (none for a call that names no agent, such as the answer forced at `max_iter`). A CrewAI that does not announce its summary of a full context window publishes that summary too: 1.15.2 makes it, unannounced, when the model refuses a request as too long.
- **What does not run in the block's context is not published.** Work CrewAI starts is included, nested crews and delegated agents among it. Work your code hands to its own threads or pools is not, wherever they were created (a Python thread does not inherit the context), unless it runs in a copy: `pool.submit(contextvars.copy_context().run, crew.kickoff)`, or `asyncio.to_thread()`, which copies it. Neither is a job handed to another process.
- **CrewAI's own model calls in a flow method are published**: `@human_feedback(llm=...)` classifies the feedback with a model call inside your method, which the adapter cannot tell from yours.
- **The sink is not a declared public API of CrewAI**, though its own frame streaming and the AG-UI bridge use it. `transport="bus"` is the fallback for a CrewAI without it: handlers on the global event bus run in a thread pool, after other listeners and out of order, so a writer thread holds each event for 50 ms and releases them in the order they were emitted (by kickoff, then by `emission_sequence`, which restarts at every top-level kickoff; kickoffs are ordered by when their first event was made), and the block's exit waits for the bus to flush. It adds latency and depends on that margin.

Not covered: serving HTTP or receiving the user's message, AG-UI's shared state and messages snapshots, resuming a paused flow, memory, knowledge, MCP and A2A events, a task's `human_input=True` (it reads the terminal and emits no event), and the AI SDK's UI message stream, for which CrewAI has no converter.

## Testing your own code

`iq_stream.testing.RecordingPublisher` records frames in memory. A crew on a fake model (a `BaseLLM` subclass) runs without network:

```python
from iq_stream.testing import RecordingPublisher
from iq_stream_crewai import publish


def test_the_crew_reaches_the_stream() -> None:
    publisher = RecordingPublisher()
    with publish(publisher) as run:
        crew.kickoff(inputs={"question": "What is the capital of France?"})
    assert publisher.outcome == "end"
    assert '"type":"RUN_FINISHED"' in publisher.data[-1]
    assert run.errors == []
```

## Requirements

- Python 3.10 to 3.13, `crewai` 1.15.2 or later (below 2), `ag-ui-protocol` 1.x, `iq-stream` 0.1.
- An IQ-Stream server 1.x.

## License

MIT
