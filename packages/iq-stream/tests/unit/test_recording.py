from __future__ import annotations

import threading

import pytest

from iq_stream import FrameTooLarge, IQError, PublishResult
from iq_stream.testing import RecordedFrame, RecordingPublisher


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


async def test_it_is_an_async_context_manager_and_close_can_be_awaited() -> None:
    async with RecordingPublisher() as pub:
        pub.send("a")
    assert pub.outcome == "end"
    other = RecordingPublisher()
    other.send("a")
    assert await other.close("fail") == PublishResult(
        200, last_event_id=1, closed=True, outcome="fail", reason="publisher"
    )
    with pytest.raises(KeyError):
        async with RecordingPublisher() as failing:
            raise KeyError
    assert failing.outcome == "fail"
