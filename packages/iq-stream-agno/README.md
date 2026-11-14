# iq-stream-agno

Publishes the runs of [Agno](https://github.com/agno-agi/agno) agents, teams and workflows to browsers through an [IQ-Stream server](https://github.com/neuron-core/iq-stream). The run keeps going when the browser leaves, the page reconnects or reloads into the same stream, and at the end it learns whether the answer finished, failed, or has to be loaded from your backend.

It is a thin layer over [`iq-stream`](https://pypi.org/project/iq-stream/), the Python client of the server. Agno hands a streaming run to the application as one iterator (`run(stream=True)`, `arun(stream=True)`); this package wraps that iterator: every event goes on to your code unchanged, becomes a frame on the stream as it arrives, and the stream is closed when the run ends.

```sh
pip install iq-stream-agno            # Agno's own event JSON
pip install "iq-stream-agno[agui]"    # AG-UI events, for CopilotKit and other AG-UI clients
pip install "iq-stream-agno[os]"      # the AgentOS interface
```

## Publish a run

```python
from iq_stream import IQClient
from iq_stream_agno import publish

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_APP_KEY, IQ_STREAM_APP_SECRET
ticket = iq.ticket()  # what the browser subscribes with: ticket.to_dict()

events = agent.run("Weather in Paris?", stream=True, stream_events=True)
for event in publish(events, iq.publisher(ticket.stream_id)):
    print(event.event)  # Agno's events, unchanged: use them as you would without IQ-Stream
# the stream is closed: "end" when the run completed, "fail" when it failed or was cancelled
```

`agent` is any Agno `Agent`, `Team` or `Workflow`; `continue_run(..., stream=True)` is wrapped the same way. In asyncio, `apublish()` wraps `arun()`:

```python
import asyncio

from iq_stream import IQClient
from iq_stream_agno import apublish

iq = IQClient()


async def answer(message: str, stream_id: str) -> None:
    events = agent.arun(message, stream=True, stream_events=True, session_id="session-1")
    async for event in apublish(events, iq.apublisher(stream_id)):
        pass  # or use the events


ticket = iq.ticket()
asyncio.run(answer("Weather in Paris?", ticket.stream_id))
```

Iterate the wrapper to the end, or close it: the stream is closed in its `finally` block, and a generator that is dropped half-way only runs it when it is garbage collected. A caller that stops early closes the stream with `fail`, and Agno stores the run as cancelled.

## Start a run from a route

`start_run()` does what a chat route needs in one call, on a running event loop: it creates the publisher (whose request opens at once, before Agno reads the session or runs its hooks), starts `entity.arun(input, stream=True, stream_events=True, ...)` in a task that publishes it, and returns without waiting.

```python
from fastapi import FastAPI
from pydantic import BaseModel

from iq_stream import IQClient
from iq_stream_agno import start_run

iq = IQClient()
app = FastAPI()


class Chat(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/chat")
async def chat(body: Chat) -> dict[str, str | int]:
    handle = start_run(iq, agent, body.message, session_id=body.session_id)
    return handle.to_dict()  # streamId, token, url, expiresAt, runId, sessionId
```

The handle has the ticket (`stream_id`, `token`, `url`, `expires_at`), the `run_id` and `session_id` it chose, and the `task`, whose result is the run's `RunSummary`: `outcome`, `status` (`completed`, `paused`, `error`, `cancelled`, `interrupted`), `run_id`, `session_id`, `frames`, `error` and the server's answer to the close line. The task never raises the run's error, which stays in the summary; cancelling it cancels the run, and the stream closes with `fail`. The stream id is recorded in the run's metadata under `iq_stream_id`. `continue_run(client, entity, run_id=..., session_id=..., requirements=...)` publishes the next leg of a paused run on a new stream. `running()` lists the tasks still publishing.

## AgentOS

```python
from agno.os import AgentOS

from iq_stream import IQClient
from iq_stream_agno.os import IQStream

agent_os = AgentOS(agents=[agent], interfaces=[IQStream(IQClient(), agent=agent)])
app = agent_os.get_app()
```

`IQStream(client, *, agent=None, team=None, workflow=None, prefix="/iq", tags=None, protocol="agno", ttl=3600, publish_options=None, shutdown_timeout=10.0)` mounts four routes, each answering with a ticket (the `start_run()` dictionary) while the run publishes from a task:

| Route | Body | |
| --- | --- | --- |
| `POST /iq/runs` | `{"message", "session_id"?, "user_id"?}` | Starts a run. |
| `POST /iq/agui` | an AG-UI `RunAgentInput` | For an agent or a team, with the `agui` extra: Agno's own AG-UI route (`run_entity()`), published instead of answered. |
| `POST /iq/runs/{run_id}/continue` | `{"session_id", "tools"?, "user_id"?}` | The next leg of a paused run, on a new stream; `tools` are the paused tools of the `RunPaused` event with the human's answers (`"confirmed": true`, ...). Needs the agent's `db`. |
| `POST /iq/runs/{run_id}/cancel` | `?user_id=` for an anonymous caller | Cancels a run this interface is publishing. |

They are gated by the same scopes as Agno's own run routes (`agents:run`, `teams:run`, `workflows:run`), and resolve the caller as those routes do: an authenticated caller runs as itself, an anonymous one may name a `user_id`. A session that belongs to another user is refused, as is continuing a run paused on an admin approval, where the installed Agno can tell. A run is continued or cancelled only by the user who started it. On shutdown the interface waits `shutdown_timeout` seconds for the runs in flight and cancels the rest, whose streams close with `fail`; agno 2.x has no shutdown hook for interfaces, so call `await interface.aclose()` from your own lifespan there.

## In the browser

The frames are unnamed and carry their own type, so a page needs one `onmessage`. What reads them is the browser side of [`@neuron-core/iq-stream`](https://www.npmjs.com/package/@neuron-core/iq-stream):

| `protocol` | A frame is | Read it with |
| --- | --- | --- |
| `"agno"` (default) | `event.to_dict()` as JSON: what AgentOS writes after `data:`, so the objects Agent UI's stream handler switches on (`{"event": "RunContent", "content": ..., "run_id": ...}`) | `frames(ticket.url, { parse: JSON.parse })` or `subscribe()` from `@neuron-core/iq-stream/client`, or a bare `EventSource` |
| `"agui"` | an AG-UI event (`{"type": "TEXT_MESSAGE_CONTENT", ...}`), from Agno's own converter | `StreamAgent` from `@neuron-core/iq-stream/ag-ui`, whose `start(input)` posts the `RunAgentInput` to your route and returns the ticket |

`frames()` closes on `iq:end` and errors on `iq:fail`, `iq:gone` and a refusal; `StreamAgent` completes after the run's `RUN_FINISHED` or `RUN_ERROR` without reporting the `iq:fail` that follows a failed run a second time. A page that receives `iq:gone` (it arrived after the close) loads the run from your backend: `agent.aget_run_output(run_id, session_id)`, or AgentOS's `GET /agents/{agent_id}/runs/{run_id}`. Both bridges were checked against what a real subscriber received (`tests/frontend/check.mjs`): the text, the tool call with its arguments and result, and the outcome.

## Which close line

| The run | The close line |
| --- | --- |
| completed | `end` |
| paused for a human (`RunPaused`): the answer goes on as a new leg on a new stream | `end` |
| the root run's `RunError`, `TeamRunError` or `WorkflowError` (Agno does not raise: it yields the error and ends) | `fail` |
| the root run's `RunCancelled`, `TeamRunCancelled` or `WorkflowCancelled` (`Agent.cancel_run(run_id)`) | `fail`, or `end` with `cancelled="end"` |
| the iterator raised, the caller stopped iterating, the task was cancelled | `fail` |
| `before_close` raised | `fail`, and the exception goes on |

Only the root run counts: its id is `root_run_id`, else the `run_id` of the first event. A team member's `RunError` inside a team that completes, or a step agent's inside a workflow that skips the step, does not fail the stream; a member's `RunCompleted` does not end it. In AG-UI the root run's ending decides the terminal event too: `RUN_FINISHED`, `RUN_ERROR`, or `RUN_FINISHED` with the cancelled outcome (`{"type": "cancelled"}`) for a cancelled or interrupted run.

**Saved before the close.** Agno stores a run before it yields the event that ends it, so an agent with a `db` has its answer saved when the stream closes, and a browser that gets `iq:gone` can load it. An agent without a `db` stores nothing: save the answer yourself in `before_close`, which is called (and awaited, for `apublish`) after the iterator ended and before the close line, with the `RunSummary`.

## Options

`publish(events, publisher, **options)` and `apublish(events, publisher, **options)` take the same options; `start_run()`, `continue_run()` and `IQStream` take them as `publish_options`.

| Option | Default | |
| --- | --- | --- |
| `protocol` | `"agno"` | `"agno"`: Agno's event JSON, lossless, for agents, teams and workflows. `"agui"`: AG-UI events (needs the `agui` extra). |
| `event_names` | `False` | Name each frame after its Agno event (or AG-UI type), as AgentOS's SSE does. A name outside the server's pattern is sent unnamed. |
| `include`, `exclude` | `None`, `()` | Agno event names to publish, or to leave out. Left-out events still decide the close line. |
| `root_run_id` | the first event's | The run whose ending decides the close line. |
| `cancelled` | `"fail"` | The close line of a cancelled run. |
| `on_oversize` | `"shrink"` | `"shrink"` an event over the frame limit (below), or `"drop"` it. |
| `transform` | `None` | Native protocol only: `transform(event)` returns the dictionary to publish, or `None` for nothing. |
| `expose_errors` | `False` | Publish error messages as they are. By default the text of an error event and of a failed tool's result is `"An error occurred."`. |
| `on_error` | log once | Called with an `IQError` for each frame that could not be published (`FrameTooLarge`, `encode_failed`, `unsupported_source`); by default each kind is logged once per run on the `iq_stream_agno` logger, without frame data. |
| `before_close` | `None` | Called with the `RunSummary` after the iterator ended by itself, before the close line. |

The publisher is the core's: `Publisher` for `publish()`, `AsyncPublisher` (or a `Publisher`) for `apublish()`. Its keepalives keep the request open while a tool runs, which Agno does in silence; an `async def` tool that blocks the event loop starves the asyncio writer, and a synchronous `Publisher` passed to `apublish()` avoids that.

## Errors and limits

Publishing fails open. An event that cannot be encoded, or a frame that does not fit, is reported and skipped; a publisher that gave up stops the encoding, and the iterator is still read to its end, so that the run and Agno's storage complete. The only exceptions that leave the wrapper are the iterator's own, and the publisher's error when the client was made with `on_error="raise"`.

**Frames over the limit** (65,536 bytes unless the app has its own). In the native protocol `"shrink"` removes members until the line fits, in this order: nested runs (`member_responses`, `step_executor_runs`, `step_results`, `step_response`), media (`images`, `videos`, `audio`, `files`, `response_audio`, `image`), context (`additional_input`, `reasoning_messages`, `references`, `session_state`); then it cuts tool results (`tool.result`, `tools[].result`) and then `content` and `reasoning_content`. What it removed or cut is listed in an added member, `iq_truncated`, which a page reads as a reason to load the full run when the stream ends. In AG-UI an oversize `TEXT_MESSAGE_CONTENT`, `TOOL_CALL_ARGS` or `REASONING_MESSAGE_CONTENT` is split into several events of its type, a `TOOL_CALL_RESULT` or a `RUN_ERROR` message is cut with ` [truncated]`, and anything else that does not fit (a `RAW` copy of a large event, a state snapshot) is dropped and reported.

- A stream holds 10 MiB by default, and native frames are a few hundred bytes each (an AG-UI text delta is about a third of a native `RunContent`): `exclude` the events your page does not read, such as `ModelRequestStarted` and `ModelRequestCompleted`. A stream the server closes for its size ends with `iq:fail` (`too_large`); the run goes on and is stored.
- `Agent.cancel_run()` reaches the process that runs the agent only through Agno's cancellation manager, in memory by default; several processes need Agno's Redis manager.
- Error text that Agno writes into a run's content, such as a workflow's "Step skipped due to error: ...", is published as Agno writes it.
- AG-UI is Agno's conversion with its limits: a team's members are not attributed, provider reasoning deltas on `RunContent` are dropped, workflow events become `RAW`. A run that ends in an error event is `RUN_ERROR` on every Agno version (agno 2.x's own converter finishes it).

## Not covered

- Agno's `background=True` runs, which yield SSE text instead of events: the wrapper passes them through, reports it, publishes nothing and closes with `fail`. `start_run()` gives the same independence from the browser's connection.
- Workflow pauses (`StepPaused`, `WorkflowPaused`) and remote agents were not run; the wrapper treats them like any other events.
- The Vercel AI SDK's UI message stream: Agno has no converter to it.

## Requirements

- Python 3.10 to 3.13, `agno` 2.9 or later below 4, `iq-stream` 0.1.
- The `agui` extra adds `agno[agui]` and FastAPI (Agno's AG-UI converter lives in `agno.os`, which imports it); the `os` extra adds `agno[os]`.
- An IQ-Stream server 1.x.

## License

MIT
