// Feeds what a real subscriber received from an IQ-Stream server to the real front-end library, and prints what the
// library made of it as one line of JSON.
//
//   IQ_STREAM_JS_DIR=/path/to/iq-stream-js node replay.mjs capture.json
//
// capture.json: {protocol, frames: [{id, event, data}], terminal: {event, data}, events: [names]}.
// protocol "ag-ui": StreamAgent of @neuron-core/iq-stream/ag-ui runs the run through @ag-ui/client's runAgent()
// (its verifier and its message reducer). protocol "crewai": the client's subscribe() with the frames' names.
// Either way the frames come from a stand-in for EventSource that replays them in order, with their ids.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const capture = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const require = createRequire(join(process.env.IQ_STREAM_JS_DIR, 'package.json'));
const load = (name) => import(pathToFileURL(require.resolve(name)).href);
const { FakeEventSource } = await load('@neuron-core/iq-stream/client/testing');

class Replay extends FakeEventSource {
  constructor(url) {
    super(url);
    setTimeout(() => {
      this.open();
      for (const { id, event, data } of capture.frames) this.frame(data, event === 'message' ? { id } : { id, event });
      this.terminal(capture.terminal.event, capture.terminal.data);
    }, 0);
  }
}

if (capture.protocol === 'ag-ui') {
  globalThis.EventSource = Replay;
  const { StreamAgent } = await load('@neuron-core/iq-stream/ag-ui');
  // start() is the application's POST; the ticket's URL is opened with the global EventSource.
  const agent = new StreamAgent({ start: async () => ({ url: 'http://iq.test/stream-id?token=sub' }) });
  const types = [];
  let outcome = null;
  let runError = null;
  let error = null;
  try {
    await agent.runAgent({}, {
      onEvent: ({ event }) => { types.push(event.type); },
      onRunFinishedEvent: (params) => { outcome = params.outcome; },
      onRunErrorEvent: ({ event }) => { outcome = 'error'; runError = { message: event.message, code: event.code }; },
    });
  } catch (failure) {
    error = String(failure?.message ?? failure);
  }
  const messages = agent.messages.map((message) => ({
    role: message.role,
    content: message.content,
    toolCalls: message.toolCalls?.map((call) => ({ name: call.function.name, arguments: call.function.arguments })),
    toolCallId: message.toolCallId,
  }));
  console.log(JSON.stringify({ types, outcome, runError, error, messages }));
} else {
  const { subscribe } = await load('@neuron-core/iq-stream/client');
  const received = [];
  const subscription = subscribe(() => new Replay('http://iq.test/stream-id'), { onFrame: (frame) => received.push(frame) }, {
    events: capture.events,
  });
  const outcome = await subscription.done;
  // What a page keeps: the text of each model call, and each tool call with its result.
  const texts = {};
  const tools = [];
  for (const frame of received) {
    const data = JSON.parse(frame.data);
    if (frame.event === 'llm_stream_chunk' && !data.tool_call) texts[data.call_id] = (texts[data.call_id] ?? '') + data.chunk;
    if (frame.event === 'tool_usage_finished') tools.push({ name: data.tool_name, args: data.tool_args, output: data.output });
    if (frame.event === 'tool_usage_error') tools.push({ name: data.tool_name, args: data.tool_args, error: data.error });
  }
  console.log(JSON.stringify({
    names: received.map((frame) => frame.event),
    ids: received.map((frame) => frame.id),
    texts: Object.values(texts),
    tools,
    outcome,
  }));
}
