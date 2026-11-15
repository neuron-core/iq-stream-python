import { frames } from '@neuron-core/iq-stream/client';
import { toUIMessageStream } from '@ai-sdk/langchain';

/** The frames of a stream, as the LangGraph parts iq-stream-langchain published: { type, ns, data }. */
export const langGraphParts = (url, options = {}) => frames(url, { ...options, parse: JSON.parse });

/** For `useStream({ transport })` of @langchain/langgraph-sdk/react: each part as LangGraph Server names it. */
export async function* useStreamEvents(parts) {
  for await (const part of parts) {
    const suffix = part.ns.length ? `|${part.ns.join('|')}` : '';
    yield { event: part.type + suffix, data: part.data };
    // The hook reads interrupts from `values`, which the adapter does not publish by default.
    if (part.type === 'updates' && part.data?.__interrupt__) {
      yield { event: `values${suffix}`, data: { __interrupt__: part.data.__interrupt__ } };
    }
    if (part.type === 'error') return; // the run said how it ended: the iq:fail after it adds nothing
  }
}

/** For `chatTransport({ open })` of @neuron-core/iq-stream/ai-sdk: UI message chunks, converted in the page. */
export function uiMessageChunks(url, options = {}) {
  let cancelled = false;
  let lost; // the subscription failed without the run saying why: iq:gone, publisher_idle, a refusal
  async function* tuples() {
    try {
      for await (const part of langGraphParts(url, options)) {
        if (part.type === 'metadata') continue;
        if (part.type === 'error') {
          cancelled = part.data.error === 'Cancelled';
          throw new Error(part.data.message);
        }
        yield part.ns.length ? [part.ns, part.type, part.data] : [part.type, part.data];
        // @ai-sdk/langchain, too, reads a human-in-the-loop interrupt from `values` only.
        if (part.type === 'updates' && part.data?.__interrupt__) {
          const values = { __interrupt__: part.data.__interrupt__ };
          yield part.ns.length ? [part.ns, 'values', values] : ['values', values];
        }
      }
    } catch (error) {
      if (error?.name === 'IQStreamError') lost = error;
      throw error;
    }
  }
  return toUIMessageStream(tuples()).pipeThrough(
    new TransformStream({
      transform(chunk, controller) {
        if (chunk.type === 'error' && lost) return;
        // A cancelled run is the AI SDK's `abort` chunk, which tells useChat it was stopped, not failed.
        controller.enqueue(chunk.type === 'error' && cancelled ? { type: 'abort' } : chunk);
      },
      flush(controller) {
        if (lost) controller.error(lost); // chatTransport reports it, as for any stream that failed
      },
    }),
  );
}
