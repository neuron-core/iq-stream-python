"""A FastAPI app that publishes LlamaIndex agent runs to IQ-Stream as AG-UI events.

``POST /chat`` starts a run and answers at once with the ticket the browser subscribes with (``StreamAgent`` from
``@neuron-core/iq-stream/ag-ui``); the run and its one request to IQ-Stream live in a task. The answer is saved before
the stream closes. ``POST /chat/{stream_id}/answer`` answers a human-input request, ``POST /chat/{stream_id}/cancel``
stops the run; both take the ticket's token, which proves the caller holds the stream.

Serve it with your own agent and store::

    uvicorn.run(create_app(agent, IQClient(), save_answer))

tests/integration/test_live_app.py serves it with a fake model and drives it over HTTP.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from iq_stream import IQClient
from pydantic import BaseModel
from workflows import Workflow
from workflows.events import Event, HumanResponseEvent, InputRequiredEvent
from workflows.handler import WorkflowHandler

from iq_stream_llamaindex import AGUIEncoder, RunOutcome, publish


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class Control(BaseModel):
    token: str  # the ticket's token


class HumanAnswer(Control):
    response: str


@dataclass
class Run:
    """What the app keeps while a run lasts: the handler to answer or cancel it, and the task that publishes it."""

    handler: WorkflowHandler
    task: asyncio.Task[RunOutcome] | None = None
    question: asyncio.Event = field(default_factory=asyncio.Event)


def create_app(agent: Workflow, iq: IQClient, save_answer: Callable[[str, RunOutcome], Awaitable[None]]) -> FastAPI:
    app = FastAPI()
    runs: dict[str, Run] = {}

    @app.post("/chat")
    async def chat(request: ChatRequest) -> dict[str, str | int]:
        ticket = iq.ticket()
        run = runs[ticket.stream_id] = Run(agent.run(user_msg=request.message, run_id=ticket.stream_id))

        def on_event(event: Event) -> None:
            if isinstance(event, InputRequiredEvent):
                run.question.set()  # the page shows it and posts the answer

        async def before_close(outcome: RunOutcome) -> None:
            await save_answer(request.thread_id, outcome)  # a browser that arrives after the close loads it from you

        run.task = asyncio.create_task(
            publish(
                run.handler,
                iq.apublisher(ticket.stream_id),
                encoder=AGUIEncoder(thread_id=request.thread_id),
                on_event=on_event,
                before_close=before_close,
            )
        )
        run.task.add_done_callback(lambda _: runs.pop(ticket.stream_id, None))
        return ticket.to_dict()

    def find(stream_id: str, token: str) -> Run:
        if not iq.verify("sub", token, stream_id):
            raise HTTPException(403, "Not your stream.")
        run = runs.get(stream_id)
        if run is None:
            raise HTTPException(404, "No run in flight for this stream.")
        return run

    @app.post("/chat/{stream_id}/answer", status_code=204)
    async def answer(stream_id: str, body: HumanAnswer) -> None:
        run = find(stream_id, body.token)
        if not run.question.is_set():
            raise HTTPException(409, "No question is waiting for an answer.")
        run.question.clear()
        run.handler.ctx.send_event(HumanResponseEvent(response=body.response))

    @app.post("/chat/{stream_id}/cancel", status_code=204)
    async def cancel(stream_id: str, body: Control) -> None:
        await find(stream_id, body.token).handler.cancel_run()

    return app
