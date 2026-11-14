from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from iq_stream import (
    AsyncPublisher,
    AsyncPublisherLike,
    FrameTooLarge,
    IQError,
    Publisher,
    PublisherLike,
    PublishResult,
)
from iq_stream.testing import AsyncRecordingPublisher, RecordedFrame, RecordingPublisher


def test_it_records_frames_and_the_first_close() -> None:
    pub = RecordingPublisher(max_frame_bytes=100)
    assert pub.send("a") is True
    assert pub.send("b", "token") is True
    assert pub.frames == [RecordedFrame("a"), RecordedFrame("b", "token")]
    assert pub.data == ["a", "b"]
    assert (pub.frames_sent, pub.bytes_sent) == (2, len('{"data":"a"}\n{"event":"token","data":"b"}\n'))
    assert pub.close("fail") == PublishResult(200, last_event_id=2, closed=True, outcome="fail", reason="publisher")
    assert pub.close("end").outcome == "fail", "the first outcome wins"
    assert pub.send("late") is False
    assert pub.close_calls == 2
    assert pub.alive is False


def test_it_validates_frames_as_the_real_publisher_does() -> None:
    pub = RecordingPublisher(max_frame_bytes=100)
    with pytest.raises(FrameTooLarge):
        pub.send("x" * 100)
    with pytest.raises(ValueError):
        pub.send("a", "iq:end")
    with pytest.raises(ValueError):
        pub.send(42)  # type: ignore[arg-type]
    assert pub.frame_size("x" * 89) == 100
    assert pub.send("x" * 89) is True
    assert pub.max_frame_bytes == 100
    assert len(pub.stream_id) == 22


def test_send_from_many_threads_records_in_the_order_the_calls_returned() -> None:
    pub = RecordingPublisher()
    threads = [threading.Thread(target=lambda i=i: [pub.send(f"{i}-{j}") for j in range(200)]) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert pub.frames_sent == 1600
    for i in range(8):
        assert [data for data in pub.data if data.startswith(f"{i}-")] == [f"{i}-{j}" for j in range(200)]


def test_fail_makes_it_inert_and_on_error_raise_makes_it_raise() -> None:
    pub = RecordingPublisher()
    pub.send("a")
    pub.fail()
    assert pub.alive is False
    assert pub.send("b") is False
    assert pub.close() == PublishResult(0, error="test")
    assert pub.error is not None and pub.error.code == "test"

    raising = RecordingPublisher(on_error="raise")
    error = IQError("IQ stream x: refused (HTTP 401, unauthorized)", status=401, code="unauthorized")
    raising.fail(error)
    with pytest.raises(IQError) as sent:
        raising.send("a")
    assert sent.value is error
    with pytest.raises(IQError):
        raising.close()


def test_abort_leaves_no_outcome() -> None:
    pub = RecordingPublisher()
    pub.abort()
    assert (pub.aborted, pub.alive, pub.close(), pub.outcome) == (True, False, PublishResult(0), None)


def test_it_is_a_sync_context_manager() -> None:
    with RecordingPublisher() as pub:
        pub.send("a")
    assert pub.outcome == "end"
    with pytest.raises(RuntimeError), RecordingPublisher() as failing:
        raise RuntimeError
    assert failing.outcome == "fail"


def test_the_synchronous_recorder_is_what_publisher_is_and_no_more() -> None:
    pub = RecordingPublisher()
    assert not hasattr(pub, "__aenter__"), "async code takes AsyncRecordingPublisher, as it takes AsyncPublisher"
    assert not inspect.isawaitable(pub.close())


async def test_the_asynchronous_recorder_is_an_async_context_manager_and_close_is_awaited() -> None:
    async with AsyncRecordingPublisher() as pub:
        pub.send("a")
    assert pub.outcome == "end"
    other = AsyncRecordingPublisher()
    other.send("a")
    assert await other.close("fail") == PublishResult(
        200, last_event_id=1, closed=True, outcome="fail", reason="publisher"
    )
    with pytest.raises(KeyError):
        async with AsyncRecordingPublisher() as failing:
            raise KeyError
    assert failing.outcome == "fail"
    assert not hasattr(failing, "__enter__")


async def test_a_close_that_is_never_awaited_records_nothing_as_with_the_real_publisher() -> None:
    pub = AsyncRecordingPublisher()
    pub.send("a")
    closing = pub.close()  # the adapter forgot the await
    assert (pub.outcome, pub.alive, pub.close_calls) == (None, True, 0)
    closing.close()  # the coroutine is dropped without the RuntimeWarning of a garbage-collected one


async def test_the_asynchronous_recorders_close_can_be_scheduled_as_a_task() -> None:
    pub = AsyncRecordingPublisher()
    assert inspect.iscoroutinefunction(pub.close)
    result = await asyncio.create_task(pub.close("fail"))
    assert (result.outcome, pub.outcome) == ("fail", "fail")


def _publish_sync(publisher: PublisherLike) -> PublishResult:
    with publisher:
        publisher.send("a")
    return publisher.close()


async def _publish_async(publisher: AsyncPublisherLike) -> PublishResult:
    async with publisher:
        publisher.send("a")
    return await publisher.close()


async def test_the_recorders_fit_the_protocols_an_adapter_types_its_publisher_with() -> None:
    sync_recorder: PublisherLike = RecordingPublisher()
    async_recorder: AsyncPublisherLike = AsyncRecordingPublisher()
    assert (await asyncio.to_thread(_publish_sync, sync_recorder)).outcome == "end"
    assert (await _publish_async(async_recorder)).outcome == "end"
    # The real publishers fit them too (the type checker says so): a function typed with one takes either.
    real: tuple[type[PublisherLike], type[AsyncPublisherLike]] = (Publisher, AsyncPublisher)
    assert real == (Publisher, AsyncPublisher)
