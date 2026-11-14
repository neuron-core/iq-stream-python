// Feeds what a real subscriber received to the browser side of @neuron-core/iq-stream and prints, as JSON, what
// that library reconstructs. Run by tests/integration/test_live_frontend.py:
//
//   IQ_STREAM_JS_DIR=/path/to/iq-stream-js node check.mjs <recording.json>
//
// The recording is {protocol, frames: [{id, event, data}], terminal: {event, data}}. Its frames are replayed into the
// client's subscription through FakeEventSource, in the order and with the ids and event names the server sent:
//   protocol "native": frames() of ./client, the transport client alone, listening to the native event names, as a
//                      page without a chat framework does;
//   protocol "agui":   StreamAgent of ./ag-ui driven by runAgent() of @ag-ui/client, as CopilotKit drives an agent.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.env.IQ_STREAM_JS_DIR;
if (!root) throw new Error('IQ_STREAM_JS_DIR is not set');
const dist = join(root, 'node_modules', '@neuron-core', 'iq-stream', 'dist');
const load = (path) => import(pathToFileURL(join(dist, path)).href);
const { frames } = await load('client/index.js');
const { FakeEventSource } = await load('client/testing.js');

const recording = JSON.parse(readFileSync(process.argv[2], 'utf8'));

// The names NativeEncoder publishes under, which a page announces to EventSource.
const NATIVE_EVENTS = [
  'RunStarted', 'AgentStream', 'AgentOutput', 'ToolCall', 'ToolCallResult', 'AgentStreamStructuredOutput',
  'InputRequiredEvent', 'StopEvent', 'WorkflowFailedEvent', 'WorkflowTimedOutEvent', 'WorkflowCancelledEvent',
  'RunAborted',
];

/** A source that replays the recording once the client listens to it. */
function replay() {
  const source = new FakeEventSource('replay');
  setImmediate(() => {
    source.open();
    for (const frame of recording.frames) {
      source.frame(frame.data, { id: frame.id, event: frame.event === 'message' ? undefined : frame.event });
    }
    source.terminal(recording.terminal.event, recording.terminal.data);
  });
  return source;
}

async function native() {
  const events = [];
  let outcome = { type: 'end' };
  const reader = frames(() => replay(), { events: NATIVE_EVENTS }).getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      events.push({ name: value.event, id: value.id, ...JSON.parse(value.data) });
    }
  } catch (error) {
    outcome = { type: error.kind, reason: error.reason ?? null };
  }
  // What a page derives from the native events: the answer, the reasoning, the tool calls, how the run ended.
  const of = (name) => events.filter((event) => event.name === name);
  const results = new Map(of('ToolCallResult').map((event) => [event.tool_id, event]));
  const last = events.at(-1) ?? null;
  return {
    events: events.length,
    ids: events.map((event) => event.id),
    text: of('AgentStream').map((event) => event.delta).join(''),
    thinking: of('AgentStream').map((event) => event.thinking_delta ?? '').join(''),
    tools: of('ToolCall').map((event) => ({
      name: event.tool_name,
      args: event.tool_kwargs,
      result: results.get(event.tool_id)?.content ?? null,
    })),
    last: last?.name ?? null,
    lastData: last,
    outcome,
  };
}

async function agui() {
  const { StreamAgent } = await load('ag-ui/index.js');
  const agent = new StreamAgent({
    threadId: 'thread-1',
    start: async () => ({ url: 'replay' }),
    open: () => frames(() => replay(), { parse: JSON.parse }),
  });
  agent.messages = [{ id: 'user-1', role: 'user', content: 'Weather in Paris?' }];
  const seen = [];
  const failures = [];
  let rejected = null;
  try {
    await agent.runAgent({}, {
      onEvent: ({ event }) => { seen.push(event); },
      onRunFailed: ({ error }) => { failures.push(String(error?.message ?? error)); },
    });
  } catch (error) {
    rejected = String(error?.message ?? error);
  }
  const assistants = agent.messages.filter((message) => message.role === 'assistant');
  return {
    events: seen.length,
    types: seen.map((event) => event.type),
    roles: agent.messages.map((message) => message.role),
    text: assistants.map((message) => message.content ?? '').join(''),
    toolCalls: assistants.flatMap((message) => message.toolCalls ?? [])
      .map((call) => ({ id: call.id, name: call.function.name, args: JSON.parse(call.function.arguments) })),
    toolResults: agent.messages.filter((message) => message.role === 'tool')
      .map((message) => ({ toolCallId: message.toolCallId, content: message.content })),
    last: seen.at(-1) ?? null,
    failures,
    rejected,
  };
}

const result = recording.protocol === 'agui' ? await agui() : await native();
process.stdout.write(JSON.stringify(result));
