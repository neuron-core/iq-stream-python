"""A Hayhooks pipeline wrapper whose run endpoint answers with the IQ-Stream ticket at once, while the agent's run is
published from a background task of the Hayhooks process.

    hayhooks run --pipelines-dir examples/hayhooks
    POST /weather_agent/run {"question": "Weather in Paris?", "thread_id": "chat-1"}
      -> {"result": {"streamId": "...", "token": "...", "url": "...", "expiresAt": ...}}

The browser subscribes to ``url``. MockChatGenerator stands for your chat generator, and ``save`` for your database.
``protocol`` lets one wrapper serve both bridges of @neuron-core/iq-stream, StreamAgent ("ag-ui") and chatTransport
("ai-sdk"); an application publishes the one its page reads.
"""

import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from hayhooks import BasePipelineWrapper
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool
from iq_stream import IQClient

from haystack_integrations.iq_stream.hayhooks import start_stream

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_PUBLIC_URL, IQ_STREAM_APP_KEY, IQ_STREAM_APP_SECRET


@tool
def get_weather(city: Annotated[str, "The city"]) -> str:
    """Get the weather for a city."""
    time.sleep(1.0)
    return f"Sunny, 21C in {city}"


def save(stream_id: str, result: dict[str, Any]) -> None:
    """Runs inside the stream's block, so the answer is stored before the close line. It may be async."""
    folder = Path(os.environ.get("ANSWERS_DIR", "answers"))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stream_id}.json").write_text(json.dumps([m.to_dict() for m in result["messages"]]))


class PipelineWrapper(BasePipelineWrapper):
    def setup(self) -> None:
        model = MockChatGenerator(
            responses=[
                ChatMessage.from_assistant(
                    text="Let me check the weather.",
                    tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Paris"}, id="call_paris")],
                ),
                "It is sunny and 21C in Paris.",
            ]
        )
        self.agent = Agent(chat_generator=model, tools=[get_weather])

    async def run_api_async(
        self,
        question: str,
        thread_id: str | None = None,
        run_id: str | None = None,
        protocol: Literal["ag-ui", "ai-sdk"] = "ag-ui",
    ) -> dict[str, Any]:
        """Starts the agent and returns what the browser needs to subscribe."""
        run_args = {"messages": [ChatMessage.from_user(question)]}
        return await start_stream(
            iq, self.agent, run_args, before_close=save, thread_id=thread_id, run_id=run_id, protocol=protocol
        )
