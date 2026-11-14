# iq-stream

The Python client of the [IQ-Stream server](https://github.com/neuron-core/iq-stream), which relays an agent's answer from a backend to browsers over Server-Sent Events. Your backend publishes the answer as it is generated; the browser reads it with `EventSource`, or with the client of [`@neuron-core/iq-stream`](https://www.npmjs.com/package/@neuron-core/iq-stream), and learns at the end whether the answer finished, failed, or has to be loaded from your backend.

It has one runtime dependency, `httpx`, and works in synchronous code, in threads and in asyncio. Adapters for LangChain, Agno, CrewAI, LlamaIndex and Haystack are built on it.

```sh
pip install iq-stream
```

## Publish an answer

```python
import json

from iq_stream import IQClient

iq = IQClient(
    url="http://iq.internal:8080",  # how this backend reaches the server
    public_url="https://stream.example.com",  # how browsers reach it
)  # app_key and app_secret from IQ_STREAM_APP_KEY, IQ_STREAM_APP_SECRET

ticket = iq.ticket()  # what the browser needs: stream_id, token, url, expires_at
# ... hand ticket.to_dict() to the browser, which subscribes to ticket.url ...

with iq.publisher(ticket.stream_id) as publisher:  # the request to the server opens now
    for chunk in run_agent(messages):
        publisher.send(json.dumps(chunk))  # one frame: queued, and written at once; never blocks
    save_answer(messages)  # before the close: a late browser loads it from you
# the block closes the stream with "end", or with "fail" if it raised
```

`run_agent()` stands for your framework's stream and `save_answer()` for your own persistence. Without arguments, `IQClient()` reads `IQ_STREAM_URL`, `IQ_STREAM_PUBLIC_URL`, `IQ_STREAM_APP_KEY` and `IQ_STREAM_APP_SECRET`.

In asyncio, `apublisher()` returns the same publisher with a task for a writer. `send()` is still a plain method: it only queues, so nothing can come between deciding to send a frame and the frame being in the queue. Only `close()` is awaited.

```python
import asyncio
import json

from iq_stream import IQClient

iq = IQClient()
running: set[asyncio.Task[None]] = set()


async def start_chat(messages: list[dict[str, str]]) -> dict[str, str | int]:
    """A route handler: starts the run and answers with the ticket at once."""
    ticket = iq.ticket()
    publisher = iq.apublisher(ticket.stream_id)  # its request opens when the loop next runs
    task = asyncio.create_task(publish(publisher, messages))
    running.add(task)
    task.add_done_callback(running.discard)
    return ticket.to_dict()  # camelCase keys, as the browser client reads them


async def publish(publisher, messages) -> None:
    async with publisher:  # "end", or "fail" on an exception or a cancellation
        async for chunk in arun_agent(messages):
            publisher.send(json.dumps(chunk))
        await asave_answer(messages)


async def main() -> None:
    ticket = await start_chat(messages)
    print("subscribe to", ticket["url"])
    await asyncio.gather(*running)


asyncio.run(main())
```

What the publisher does for you:

- **One request per answer.** It opens a single chunked `POST` when it is created and holds it to the close line, which is what the server is built for. A frame is one line of it.
- **Silence is covered.** While a tool runs it writes a keepalive line every 20 seconds, so the server's read timeout does not let the request go.
- **A lost request is continued.** After a `408`, a `503` or a dropped connection it asks the server how many frames it holds and resends exactly the missing ones. If the gap cannot be bridged, as after a server restart, it closes the stream with `fail` and stops, so that no browser gets an answer with a hole in it.
- **Publishing never breaks the run.** By default an error on the way to the server is reported once, the publisher goes inert, and `send()` returns `False`. Your agent finishes and saves its answer.
- **A quick answer is not lost.** The close line is held back until the stream is three seconds old (`min_open`), so a browser that subscribes just after a short answer still receives it. Frames are never delayed.
- **A cancelled run is closed.** The async close is shielded from cancellation, so a cancelled task still sends `fail` and its browser hears of it at once. Synchronous publishers still open when the interpreter exits are closed with `fail` by an `atexit` hook.

## Handing the stream to the browser

A browser needs the stream id, a `sub` token and the server's public address. `iq.ticket(stream_id=None, ttl=3600)` returns them as a frozen `Ticket`: `stream_id`, `token`, `url` (the subscribe URL with the token in its query, ready for `new EventSource(url)`) and `expires_at` in Unix seconds. `ticket.to_dict()` has the camelCase keys (`streamId`, `token`, `url`, `expiresAt`) that the browser client's `chatTransport`, `StreamAgent` and `subscribe` take.

There are two orders, and both work:

- **Start, then ticket.** The request that starts the run creates the publisher and answers with the ticket, as `start_chat()` does above. One round trip; the server replays a running stream from its first frame to whoever subscribes, and `min_open` covers answers that finish before the browser arrives.
- **Ticket, subscribe, then start.** The page asks for a ticket, opens its `EventSource`, waits for `open`, then asks for the run with the stream id and the token. The backend checks that the id came from it:

```python
from iq_stream import IQClient

iq = IQClient()
ticket = iq.ticket()  # the first request: the page subscribes with it


def start_run(stream_id: str, token: str, messages: list[dict[str, str]]) -> None:
    """The second request: the page sends back the stream id and its token."""
    if not iq.verify("sub", token, stream_id):
        raise PermissionError("this stream id was not handed out here")
    with iq.publisher(stream_id) as publisher:
        for chunk in run_agent(messages):
            publisher.send(chunk["delta"])


start_run(ticket.stream_id, ticket.token, messages)
```

When the run waits in a job queue, call `iq.reserve(stream_id)` where the ticket is handed out: a subscriber waits only 30 seconds for a publisher, and the reservation makes it wait the stream's idle timeout (two minutes by default) instead. A process that never had the publisher, such as a job's failure handler after the worker died, closes the stream with `iq.close_stream(stream_id)`, so that subscribers get `iq:fail` at once. `areserve()` and `aclose_stream()` are the same for asyncio.

```python
from iq_stream import IQClient

iq = IQClient()
ticket = iq.ticket()
iq.reserve(ticket.stream_id)  # the job may start minutes from now
# ... the worker dies before it publishes anything ...
result = iq.close_stream(ticket.stream_id)  # "fail" by default; a closed stream is left as it is
assert result.closed and result.outcome == "fail"
```

**Which close line.** `end` when the answer finished, `fail` when it did not, a cancelled run included: a second tab watching the stream should not take a partial answer for a finished one. Send your protocol's own final event (AG-UI's `RUN_FINISHED`, the AI SDK's `finish` or `abort` chunk) as a frame before the close line; it is what tells the page why. Close with `end` only after the answer is saved, since a browser that arrives after the close receives `iq:gone` and loads the answer from you.

## Reference

### `IQClient`

```text
IQClient(url=None, app_key=None, app_secret=None, *, public_url=None, keepalive=20.0, min_open=3.0,
         max_frame_bytes=65_536, on_error="report", error_handler=None, retry_budget=10.0, takeover_budget=65.0,
         replay_frames=512, replay_bytes=4 MiB, queue_bytes=8 MiB, trust_env=None, verify=True)
```

| Method | |
| --- | --- |
| `publisher(stream_id)` | The publisher of one stream for synchronous code: a daemon thread writes. Its request opens at once. |
| `apublisher(stream_id)` | The same for asyncio, called on a running loop: a task writes. |
| `ticket(stream_id=None, ttl=3600)` | What the browser needs. Without a stream id, a new one is made. |
| `new_stream_id()` | 128 random bits as base64url. Never reuse an id for a second answer. Also `iq_stream.new_stream_id()`. |
| `token(kind, stream_id, ttl=3600)`, `subscribe_url(stream_id, ttl=3600)` | The parts of a ticket. The server refuses a `ttl` beyond its `MAX_TOKEN_TTL_S` (3600 by default). |
| `verify(kind, token, stream_id)` | Whether a token is one this backend minted for that stream and is still valid. |
| `reserve(stream_id)`, `areserve(stream_id)` | One empty request that takes the stream out of `pending`. |
| `close_stream(stream_id, outcome="fail")`, `aclose_stream(...)` | Closes a stream from a process that never had its publisher. |

The client holds no connection and no thread until a publisher or a request needs one, so a worker pool may create it before it forks. It is given the app secret: it belongs to your backend, never to a browser, and the secret never appears in its `repr`, in an error or in a log line.

### `Publisher` and `AsyncPublisher`

| Member | |
| --- | --- |
| `send(data, event=None) -> bool` | Queues one frame and returns whether the publisher is alive. `data` is a string, usually JSON; `event` an optional SSE event name. Raises `FrameTooLarge` for a line over `max_frame_bytes` and `ValueError` for data that is not a string or an invalid event name. Never waits for the network. |
| `close(outcome="end", timeout=10.0) -> PublishResult` | Sends the close line and returns the server's answer; awaited, and shielded from cancellation, in `AsyncPublisher`. The first outcome wins and every call returns the same result. `timeout` counts from the moment the close line may go out. |
| `abort()` | Drops the connection and leaves the stream open, for a process that hands the stream to another. |
| `with` / `async with` | Closes with `end` on a normal exit, and with `fail` on an exception, a `KeyboardInterrupt` or a cancellation, which then goes on. |
| `frame_size(data, event=None)`, `max_frame_bytes` | The size a frame would have on the wire, envelope and escapes included, and the limit it must fit. |
| `frames_sent`, `bytes_sent`, `alive`, `error`, `requests`, `stream_id` | What was queued, whether the publisher gave up and why, and how many requests the stream took. |

A synchronous `Publisher` may be created, used and closed inside a running event loop: `send()` only touches a thread-safe queue. Its `close()` waits for the server, so async code calls it through `asyncio.to_thread`, and it may be called from any thread. `send()` may be called from any number of threads; frames go out in the order the calls returned. `AsyncPublisher.send()` belongs to its loop's thread: a framework that emits from worker threads takes the synchronous publisher.

`PublishResult` has `status` (0 when the publisher had no answer), `last_event_id` (the frames the stream holds), `closed`, `outcome` (`"end"` or `"fail"`), `reason` (why the stream failed) and `error` (the server's code, or the publisher's own: `gap`, `queue_full`, `close_timeout`, `unreachable`, `cancelled`).

### Options

| Option | Default | |
| --- | --- | --- |
| `public_url` | `IQ_STREAM_PUBLIC_URL`, else `url` | The address in tickets, when browsers reach the server elsewhere than this backend does. |
| `keepalive` | `20.0` | Seconds of silence after which an empty line is written. `None` turns keepalives off. |
| `min_open` | `3.0` | Seconds a stream stays open before its close line goes out. |
| `max_frame_bytes` | `65_536` | The limit one line must fit, measured as the server measures it. Lower it to the app's own limit, if it has one. |
| `on_error` | `"report"` | `"report"`: an error is handed to `error_handler` (else logged once as a warning on the `iq_stream` logger) and the publisher goes inert. `"raise"`: the next `send()` or `close()` raises it. |
| `error_handler` | `None` | Called once per publisher with the `IQError` it gave up on. |
| `retry_budget` | `10.0` | Seconds spent retrying a `503`, `502`, `504` or a refused connection. |
| `takeover_budget` | `65.0` | Seconds spent waiting for the server to let go of a lost request of ours (`409 publisher_active`), which it does after its read timeout. |
| `replay_frames`, `replay_bytes` | `512`, 4 MiB | What is kept to resend after a lost request. |
| `queue_bytes` | 8 MiB | Bytes that may wait for the server before the stream is given up. |
| `trust_env` | auto | Whether httpx reads proxy settings from the environment. Off for loopback and private addresses and single-label host names, since a proxy would buffer the request body; `IQ_STREAM_TRUST_ENV=1` turns it on, `=0` off. |
| `verify` | `True` | TLS verification, or an `ssl.SSLContext`. |

`url` may be `http://`, `https://` (with a path prefix if the server sits behind one), or `http+unix:///path/to/iq.sock` for a server started with `SOCKET=`. Durations are seconds.

## Errors

Every error is an `IQError` with `stream_id`, `status` (the HTTP status, when there was an answer) and `code` (the server's error code or the publisher's own), and a message that never carries a token, the secret or frame data. Configuration mistakes raise `IQError` from the constructor.

Publishing fails open. A publisher that gives up, on a refusal or after its retries, closes the stream with `fail` when the server still holds it, reports the error once, and goes inert: `send()` returns `False`, `alive` is `False` and `error` says why. An adapter reads `alive` to stop encoding frames nobody will receive, while it goes on draining the framework's stream so that the run and its persistence complete.

`FrameTooLarge` (an `IQError` with `size` and `limit`) is raised by `send()` itself, at once, because the caller is the one who knows what to send instead. The publisher does not split frames, since only the protocol knows how a client puts them back together:

```python
import json

from iq_stream import FrameTooLarge, IQClient

iq = IQClient(min_open=0)
tool_result = "x" * 100_000

with iq.publisher(iq.new_stream_id()) as publisher:
    event = {"type": "TOOL_CALL_RESULT", "content": tool_result}
    try:
        publisher.send(json.dumps(event))
    except FrameTooLarge as error:
        room = len(tool_result) - (error.size - publisher.max_frame_bytes) - 100
        event["content"] = tool_result[:room] + " [truncated]"
        publisher.send(json.dumps(event))
```

## Limits worth knowing

- A frame is at most 65,536 bytes and a stream at most 10 MiB and 30 minutes, unless the app has limits of its own on the server.
- httpx sends a request's whole body before it reads the answer. The publisher therefore looks at its connection's socket before every write, and every second while it waits, and stops writing as soon as the server has answered or dropped it. On TLS connections it cannot (TLS records arrive unasked), and an answer that comes while nothing is written is seen at the next write, keepalives included; the counting of frames makes up for what was written meanwhile.
- A connection that stalls without being reset is noticed when the queue of unsent frames reaches `queue_bytes`, or when `close()` times out.
- The writer of a synchronous publisher is a daemon thread: call `close()` (or use the `with` block) before the process exits. The `atexit` hook closes what is left with `fail`; a process killed outright closes nothing, and its subscribers get `iq:fail` with `publisher_idle` after the idle timeout.
- A stream belongs to the publisher that started it: another process cannot continue it after `abort()`.

## Testing your own code

`iq_stream.testing` ships with the package and needs nothing beyond the standard library and httpx.

`RecordingPublisher` has the publisher's surface in memory, for unit tests without a network. It validates frames as the real one does, records them (`frames`, `data`) from any number of threads in the order the calls returned, records the close `outcome` (the first one wins), works as a sync and as an async context manager, and its `close()` can be called or awaited, so it stands in for both publishers. `fail()` makes it give up as the real one does after a fatal answer; with `on_error="raise"` it then raises.

```python
import json

from iq_stream.testing import RecordingPublisher


def publish_run(publisher, chunks):  # the adapter under test
    with publisher:
        for chunk in chunks:
            if not publisher.send(json.dumps(chunk)):
                break


def test_each_chunk_is_a_frame_and_the_run_ends() -> None:
    publisher = RecordingPublisher()
    publish_run(publisher, [{"delta": "Hi"}, {"delta": "!"}])
    assert publisher.data == ['{"delta": "Hi"}', '{"delta": "!"}']
    assert publisher.outcome == "end"


def test_publishing_fails_open() -> None:
    publisher = RecordingPublisher()
    publisher.fail()  # as after a fatal answer from the server
    publish_run(publisher, [{"delta": "Hi"}])
    assert publisher.data == []
```

For integration tests, the kit finds two IQ-Stream servers: one with default settings and one with short timeouts and small limits (`SHORT_SETTINGS`). Each is the one `IQ_STREAM_TEST_URL` or `IQ_STREAM_TEST_SHORT_URL` names, else the one answering on `http://127.0.0.1:8080` or `:8081`, else a container started from `IQ_STREAM_TEST_IMAGE` (default `ghcr.io/neuron-core/iq-stream:1`). The pytest fixtures come from a plugin that a suite opts into in its `conftest.py`:

```python
# conftest.py
pytest_plugins = ["iq_stream.testing.pytest_plugin"]
```

`iq_servers` (one per session) has `default.url`, `short.url`, `app_key` and `app_secret`; `iq_client` and `iq_short_client` are clients of each, with `min_open=0`. `subscribe()` reads a stream as a browser would, up to its terminal event, and `subscribe_in_background()` starts that before the publisher and is collected after it, with `result()` or `await`:

```python
from iq_stream.testing import subscribe_in_background


def test_a_run_reaches_the_browser(iq_client, iq_servers) -> None:
    ticket = iq_client.ticket()
    subscription = subscribe_in_background(iq_servers.default.url, ticket.stream_id, ticket.token)
    with iq_client.publisher(ticket.stream_id) as publisher:
        publisher.send("Hi")
        publisher.send("there", event="token")
    received = subscription.result()
    assert received.data == ["Hi", "there"]
    assert [frame.event for frame in received.frames] == ["message", "token"]
    assert received.terminal.event == "iq:end"
```

A `Subscription` has `frames` (each with `id`, `event`, `data` and `at`, its arrival in seconds), `terminal` (`event` and its parsed `data`) and `status`. `asubscribe()` is the same for asyncio. `start_server()` starts a server of the test's own, on a free port, with `kill()`, `start()` and `stop()`, for tests that crash or restart it.

## Requirements

- Python 3.10 to 3.13, `httpx` 0.27 or later.
- An IQ-Stream server 1.x.

## License

MIT
