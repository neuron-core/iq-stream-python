"""The answer is saved before the close line: by the checkpointer when the iterator ends, by the application after
publish_run() returns, inside the publisher's block."""

from __future__ import annotations

import asyncio
from typing import Any

from fakes import FINAL, USER, weather_agent
from iq_stream import Outcome, PublishResult
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langgraph.checkpoint.memory import InMemorySaver

from iq_stream_langchain import apublish_run, publish_run


class LoggingSaver(InMemorySaver):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        result = super().put(config, checkpoint, metadata, new_versions)
        self.log.append(f"put step={metadata.get('step')}")
        return result


class SlowSaver(InMemorySaver):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        await asyncio.sleep(0.25)
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self.log.append(f"aput step={metadata.get('step')}")
        return result


class LoggingPublisher(RecordingPublisher):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        self.log.append(f"close {outcome}")
        return super().close(outcome, timeout)


class AsyncLoggingPublisher(AsyncRecordingPublisher):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self.log = log

    async def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        self.log.append(f"close {outcome}")
        return await super().close(outcome, timeout)


def test_the_last_checkpoint_and_the_application_save_come_before_the_close() -> None:
    log: list[str] = []
    agent = weather_agent(checkpointer=LoggingSaver(log))
    config: Any = {"configurable": {"thread_id": "t1"}}
    publisher = LoggingPublisher(log)
    with publisher:
        final = publish_run(publisher, agent, USER, config)
        log.append("saved")
    assert log[-3:] == ["put step=3", "saved", "close end"]
    assert final["messages"][-1].content == FINAL
    assert agent.get_state(config).values["messages"][-1].content == FINAL


async def test_a_slow_async_checkpointer_finishes_before_the_close() -> None:
    log: list[str] = []
    agent = weather_agent(checkpointer=SlowSaver(log))
    publisher = AsyncLoggingPublisher(log)
    async with publisher:
        await apublish_run(publisher, agent, USER, {"configurable": {"thread_id": "t2"}})  # durability "async"
    assert log[-2:] == ["aput step=3", "close end"]
