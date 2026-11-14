"""The writer's state machine against a scripted transport: each row of core-client.md's table of answers, and what
the writer does next. Every test runs for Publisher and for AsyncPublisher (the ``kind`` fixture)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import pytest
from fakes import SECRET, STREAM, Harness, Pending, answer, closed_fail, open_, until

from iq_stream import FrameTooLarge, IQError, PublishResult

Setup = Callable[..., Harness]


def result(status: int, **fields: object) -> PublishResult:
    return PublishResult(status=status, **fields)  # type: ignore[arg-type]


async def test_the_request_opens_at_creation_lines_go_out_in_order_and_close_returns_the_answer(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    request = await h.request(0)
    assert request.lines == [], "opened before any frame"
    assert request.stream_id == STREAM
    assert h.client.verify("pub", request.token, STREAM), "a pub token for this stream"

    assert pub.send("a") is True
    assert pub.send("b", "token") is True
    assert pub.send('{"type":"x"}') is True
    await until(lambda: len(request.lines) == 3)
    assert request.lines == ['{"data":"a"}', '{"event":"token","data":"b"}', '{"data":"{\\"type\\":\\"x\\"}"}']
    assert pub.alive is True

    closing = asyncio.ensure_future(h.close(pub))
    await until(lambda: request.ended)
    assert request.closes == ["end"]
    assert pub.alive is False
    assert pub.send("late") is False, "nothing is queued after close()"
    request.respond(200, stream=STREAM, lastEventId=3, closed=True, outcome="end")
    assert await closing == result(200, last_event_id=3, closed=True, outcome="end")

    assert pub.requests == 1
    assert pub.frames_sent == 3
    assert pub.bytes_sent == len("\n".join(request.lines[:3])) + 1, "the bytes of the lines queued, newlines included"
    assert pub.error is None
    assert h.transport.posts == []
    assert h.errors == []


async def test_408_reopens_and_resumes_from_last_event_id_without_a_probe(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    for data in "abcd":
        pub.send(data)
    first = await h.request(0)
    await until(lambda: len(first.data) == 4)
    first.respond(408, lastEventId=2, closed=False, error="read_timeout")

    second = await h.request(1)
    await until(lambda: len(second.data) == 2)
    assert second.data == ["c", "d"], "exactly the frames the server does not hold"
    pub.send("e")
    await until(lambda: len(second.data) == 3)
    second.respond(408, lastEventId=5, closed=False, error="read_timeout")

    third = await h.request(2)
    closing = asyncio.ensure_future(h.close(pub))
    await until(lambda: third.ended)
    assert third.lines == ['{"close":"end"}'], "nothing to resend"
    third.respond(200, lastEventId=5, closed=True, outcome="end")
    assert (await closing).last_event_id == 5
    assert pub.requests == 3
    assert h.transport.probes == 0
    assert h.errors == []


async def test_an_answer_that_arrives_while_the_request_is_quiet_is_seen_without_a_write(setup: Setup) -> None:
    h = setup(keepalive=None)
    pub = h.publisher()
    pub.send("a")
    first = await h.request(0)
    await until(lambda: len(first.data) == 1)
    first.respond(408, lastEventId=1, closed=False, error="read_timeout")
    second = await h.request(1)
    assert second.lines == []
    pub.send("b")
    await until(lambda: second.data == ["b"])
    second.respond(200, lastEventId=2, closed=True, outcome="end")
    assert (await h.close(pub)).last_event_id == 2


async def test_a_lost_connection_is_probed_and_what_the_server_is_missing_is_resent(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    for i in range(1, 6):
        pub.send(f"f{i}")
    first = await h.request(0)
    await until(lambda: len(first.data) == 5)
    h.transport.on_post = lambda body, n: open_(3)
    first.respond(0)

    second = await h.request(1)
    await until(lambda: len(second.data) == 2)
    assert h.transport.posts == [""], "one probe, with an empty body"
    assert second.data == ["f4", "f5"]
    pub.send("f6")
    pub.send("f7")
    closing = asyncio.ensure_future(h.close(pub))
    await until(lambda: second.ended)
    assert second.data == ["f4", "f5", "f6", "f7"]
    second.respond(200, lastEventId=7, closed=True, outcome="end")
    assert await closing == result(200, last_event_id=7, closed=True, outcome="end")
    assert h.errors == []


async def test_frames_sent_during_a_recovery_wait_in_the_queue_and_follow_the_resent_ones(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    first = await h.request(0)
    await until(lambda: len(first.data) == 1)
    probe = Pending()
    h.transport.on_post = lambda body, n: probe
    first.respond(0)
    await until(lambda: h.transport.probes == 1)
    pub.send("b")
    pub.send("c")
    probe.resolve(open_(0))
    second = await h.request(1)
    await until(lambda: len(second.data) == 3)
    assert second.data == ["a", "b", "c"]
    second.respond(200, lastEventId=3, closed=True, outcome="end")
    await h.close(pub)


async def test_nothing_is_written_into_a_request_the_server_has_answered(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    first = await h.request(0)
    await until(lambda: len(first.data) == 1)
    first.watch.settle()  # the head of an answer has arrived; its body has not
    pub.send("b")
    pub.send("c")
    await asyncio.sleep(0.05)
    assert first.data == ["a"]
    first.respond(408, lastEventId=1, closed=False, error="read_timeout")
    second = await h.request(1)
    await until(lambda: len(second.data) == 2)
    assert second.data == ["b", "c"]
    second.respond(200, lastEventId=3, closed=True, outcome="end")
    await h.close(pub)


async def test_a_close_line_lost_with_its_request_is_sent_again(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    closing = asyncio.ensure_future(h.close(pub, "fail"))
    first = await h.request(0)
    await until(lambda: first.ended)
    first.respond(408, lastEventId=1, closed=False, error="read_timeout")
    second = await h.request(1)
    await until(lambda: second.ended)
    assert second.lines == ['{"close":"fail"}']
    second.respond(200, lastEventId=1, closed=True, outcome="fail", reason="publisher")
    assert await closing == result(200, last_event_id=1, closed=True, outcome="fail", reason="publisher")


async def test_a_close_line_that_arrived_when_its_answer_was_lost_is_the_publishers_own_close(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    request = await h.request(0)
    closing = asyncio.ensure_future(h.close(pub, "end"))
    await until(lambda: request.ended)
    h.transport.on_post = lambda body, n: answer(409, error="stream_closed", outcome="end")
    request.respond(0)
    assert await closing == result(409, closed=True, outcome="end", error="stream_closed")
    assert h.errors == []
    assert pub.error is None
    assert h.transport.closes == [], "nothing is closed out of band"


async def test_a_stream_that_closed_otherwise_while_the_close_line_was_on_its_way_is_an_error(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    request = await h.request(0)
    closing = asyncio.ensure_future(h.close(pub, "fail"))
    await until(lambda: request.ended)
    h.transport.on_post = lambda body, n: answer(409, error="stream_closed", outcome="fail", reason="max_duration")
    request.respond(0)
    assert (await closing).reason == "max_duration"
    assert len(h.errors) == 1
    assert h.errors[0].code == "stream_closed"


async def test_409_publisher_active_on_the_probe_is_retried_with_backoff(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    first = await h.request(0)
    await until(lambda: len(first.data) == 1)
    h.transport.on_post = lambda body, n: answer(409, error="publisher_active") if n < 2 else open_(1)
    started = time.monotonic()
    first.respond(0)
    second = await h.request(1)
    assert h.transport.probes == 3
    assert time.monotonic() - started >= 0.25, "the retries are spaced out"
    assert second.data == []
    second.respond(200, lastEventId=1, closed=True, outcome="end")
    await h.close(pub)
    assert h.errors == []


async def test_409_publisher_active_may_outlast_retry_budget(setup: Setup) -> None:
    h = setup(retry_budget=0.2, takeover_budget=5)
    pub = h.publisher()
    pub.send("a")
    first = await h.request(0)
    await until(lambda: len(first.data) == 1)
    started = time.monotonic()
    h.transport.on_post = lambda body, n: (
        answer(409, error="publisher_active") if time.monotonic() - started < 0.6 else open_(1)
    )
    first.respond(0)
    pub.send("b")
    second = await h.request(1)
    await until(lambda: len(second.data) == 1)
    assert time.monotonic() - started >= 0.6, "it waited well past retry_budget"
    assert second.data == ["b"]
    second.respond(200, lastEventId=2, closed=True, outcome="end")
    assert (await h.close(pub)).status == 200
    assert h.errors == []


async def test_409_publisher_active_that_never_clears_stops_at_takeover_budget_and_closes_with_fail(
    setup: Setup,
) -> None:
    h = setup(takeover_budget=0.5)
    pub = h.publisher()
    pub.send("a")
    h.transport.on_post = lambda body, n: answer(409, error="publisher_active") if body == "" else closed_fail(0)
    started = time.monotonic()
    (await h.request(0)).respond(409, error="publisher_active")
    outcome = await h.close(pub)
    assert time.monotonic() - started < 1.5, "bounded"
    assert h.transport.probes >= 2, "retried"
    assert h.transport.closes == ["fail"]
    assert outcome == result(
        409, closed=True, error="publisher_active", last_event_id=0, outcome="fail", reason="publisher"
    )
    assert len(h.errors) == 1
    assert (h.errors[0].status, h.errors[0].code) == (409, "publisher_active")
    assert pub.requests == 1, "no second stream request while the server says 'not now'"


@pytest.mark.parametrize("status", [503, 502, 504])
async def test_503_502_504_are_retried_with_backoff_and_the_stream_goes_on(setup: Setup, status: int) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    pub.send("b")
    h.transport.on_post = lambda body, n: answer(status, error="unavailable") if n == 0 else open_(0)
    (await h.request(0)).respond(status, error="unavailable")
    second = await h.request(1)
    await until(lambda: len(second.data) == 2)
    assert h.transport.probes == 2
    assert second.data == ["a", "b"]
    second.respond(200, lastEventId=2, closed=True, outcome="end")
    assert (await h.close(pub)).status == 200
    assert h.errors == []


async def test_frames_sent_during_an_outage_do_not_cut_the_backoff_between_probes_short(setup: Setup) -> None:
    h = setup(retry_budget=5)
    h.transport.on_open = lambda request, n: request.respond(0)
    h.transport.on_post = lambda body, n: answer(503, error="unavailable")
    pub = h.publisher()
    deadline = time.monotonic() + 0.7
    while time.monotonic() < deadline:
        pub.send("token")
        await asyncio.sleep(0.005)
    assert h.transport.probes <= 4, f"{h.transport.probes} probes in 700 ms"  # at 0, 200 and 600 ms
    pub.abort()


async def test_503_that_lasts_is_fatal_after_retry_budget(setup: Setup) -> None:
    h = setup(retry_budget=0.4)
    pub = h.publisher()
    h.transport.on_post = lambda body, n: answer(503, error="unavailable")
    (await h.request(0)).respond(503, error="unavailable")
    await until(lambda: not pub.alive)
    assert pub.send("a") is False
    outcome = await h.close(pub)
    assert outcome.status == 503
    assert outcome.closed is False
    assert pub.error is h.errors[0]
    assert pub.error.status == 503
    assert h.transport.probes >= 1


async def test_no_answer_at_all_is_retried_within_retry_budget_then_given_up(setup: Setup) -> None:
    h = setup(retry_budget=0.4)
    h.transport.on_post = lambda body, n: answer(0)
    h.transport.on_open = lambda request, n: request.respond(0)
    pub = h.publisher()
    assert await h.close(pub) == result(0, closed=False, error="unreachable")
    assert h.transport.probes >= 2
    assert len(h.errors) == 1
    assert h.errors[0].status is None
    assert h.errors[0].code == "unreachable"


async def test_a_stream_request_that_keeps_failing_while_probes_succeed_does_not_loop_for_ever(setup: Setup) -> None:
    h = setup(retry_budget=0.4)
    h.transport.on_open = lambda request, n: request.respond(0)
    pub = h.publisher()
    pub.send("a")
    outcome = await h.close(pub)
    assert outcome.error == "unreachable"
    assert outcome.outcome == "fail", "the stream the probes created is closed"
    assert 2 <= pub.requests < 10, f"{pub.requests} requests"
    assert len(h.errors) == 1


async def test_409_stream_closed_is_fatal_and_there_is_nothing_to_close(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    (await h.request(0)).respond(409, error="stream_closed", outcome="fail", reason="max_duration")
    await until(lambda: not pub.alive)
    assert pub.send("b") is False
    assert await h.close(pub) == result(409, closed=True, outcome="fail", reason="max_duration", error="stream_closed")
    assert h.transport.posts == []
    assert [(e.code, e.stream_id) for e in h.errors] == [("stream_closed", STREAM)]
    assert pub.requests == 1


async def test_413_stream_too_large_means_the_server_closed_the_stream(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    (await h.request(0)).respond(
        413, lastEventId=9, closed=True, outcome="fail", reason="too_large", error="stream_too_large"
    )
    assert await h.close(pub) == result(
        413, last_event_id=9, closed=True, outcome="fail", reason="too_large", error="stream_too_large"
    )
    assert h.transport.posts == []
    assert h.errors[0].code == "stream_too_large"


@pytest.mark.parametrize(("status", "error"), [(413, "frame_too_large"), (400, "invalid_frame")])
async def test_a_refused_line_leaves_the_stream_open_so_it_is_closed_with_fail_out_of_band(
    setup: Setup, status: int, error: str
) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("ok")
    pub.send("refused")
    h.transport.on_post = lambda body, n: closed_fail(1)
    request = await h.request(0)
    await until(lambda: len(request.data) == 2)
    request.respond(status, lastEventId=1, closed=False, error=error, line=2)
    await until(lambda: not pub.alive)
    assert pub.send("after") is False
    assert await h.close(pub, "end") == result(
        status, last_event_id=1, closed=True, outcome="fail", reason="publisher", error=error
    )
    assert h.transport.posts == ['{"close":"fail"}\n']
    assert [e.code for e in h.errors] == [error]
    assert pub.requests == 1


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, "unauthorized"), (403, "forbidden"), (415, "unsupported_media_type"), (429, "app_limit"), (404, None)],
)
async def test_fatal_answers_where_the_stream_may_not_exist_close_nothing(
    setup: Setup, status: int, error: str | None
) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    body = {} if error is None else {"error": error}
    (await h.request(0)).respond(status, **body)
    assert await h.close(pub) == result(status, closed=False, error=error)
    assert h.transport.posts == []
    assert [(e.status, e.code) for e in h.errors] == [(status, error)]
    assert pub.alive is False
    assert pub.requests == 1


async def test_an_error_says_which_stream_and_why_and_never_carries_a_token_the_secret_or_frame_data(
    setup: Setup,
) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("very-private-frame-data")
    request = await h.request(0)
    await until(lambda: len(request.data) == 1)
    request.respond(413, lastEventId=0, closed=False, error="frame_too_large", line=1)
    await h.close(pub)
    [error] = h.errors
    assert isinstance(error, IQError)
    assert STREAM in str(error)
    assert "413" in str(error)
    assert "frame_too_large" in str(error)
    everything = f"{error} {error!r} {vars(error)}"
    for secret in ("very-private-frame-data", SECRET, request.token):
        assert secret not in everything


async def test_the_replay_ring_is_bounded_by_frames_and_a_gap_it_cannot_bridge_closes_with_fail(setup: Setup) -> None:
    h = setup(replay_frames=2)
    pub = h.publisher()
    for i in range(1, 6):
        pub.send(f"f{i}")
    h.transport.on_post = lambda body, n: closed_fail(2)
    request = await h.request(0)
    await until(lambda: len(request.data) == 5)
    request.respond(408, lastEventId=2, closed=False, error="read_timeout")
    assert await h.close(pub, "end") == result(
        408, last_event_id=2, closed=True, outcome="fail", reason="publisher", error="gap"
    )
    assert h.transport.posts == ['{"close":"fail"}\n']
    assert pub.requests == 1, "a stream with a hole in it is not continued"
    assert h.errors[0].code == "gap"


async def test_the_replay_ring_holds_exactly_replay_frames(setup: Setup) -> None:
    h = setup(replay_frames=2)
    pub = h.publisher()
    for i in range(1, 6):
        pub.send(f"f{i}")
    first = await h.request(0)
    await until(lambda: len(first.data) == 5)
    first.respond(408, lastEventId=3, closed=False, error="read_timeout")
    second = await h.request(1)
    await until(lambda: len(second.data) == 2)
    assert second.data == ["f4", "f5"]
    second.respond(200, lastEventId=5, closed=True, outcome="end")
    await h.close(pub)
    assert h.errors == []


async def test_the_replay_ring_is_bounded_by_bytes_too(setup: Setup) -> None:
    h = setup(replay_bytes=250)
    pub = h.publisher()
    for i in range(1, 5):
        pub.send(str(i) * 89)  # each line is 101 bytes: the ring holds two
    first = await h.request(0)
    await until(lambda: len(first.data) == 4)
    first.respond(408, lastEventId=2, closed=False, error="read_timeout")
    second = await h.request(1)
    await until(lambda: len(second.data) == 2)
    assert [data[0] for data in second.data] == ["3", "4"]
    second.respond(408, lastEventId=1, closed=False, error="read_timeout")
    await until(lambda: not pub.alive)
    assert h.errors[0].code == "gap"
    await h.close(pub)


async def test_a_restarted_server_reports_fewer_frames_than_the_ring_can_bridge(setup: Setup) -> None:
    h = setup(replay_frames=3)
    pub = h.publisher()
    for i in range(1, 9):
        pub.send(f"f{i}")
    first = await h.request(0)
    await until(lambda: len(first.data) == 8)
    h.transport.on_post = lambda body, n: open_(0) if body == "" else closed_fail(0)
    first.respond(0)
    outcome = await h.close(pub)
    assert (outcome.outcome, outcome.error) == ("fail", "gap")
    assert h.transport.posts == ["", '{"close":"fail"}\n'], "nothing is published into the new stream"
    assert len(h.transport.requests) == 1
    assert len(h.errors) == 1


async def test_after_a_restart_an_answer_the_ring_holds_whole_is_published_again_from_its_first_frame(
    setup: Setup,
) -> None:
    h = setup()
    pub = h.publisher()
    for i in range(1, 4):
        pub.send(f"f{i}")
    first = await h.request(0)
    await until(lambda: len(first.data) == 3)
    h.transport.on_post = lambda body, n: open_(0)
    first.respond(0)
    second = await h.request(1)
    await until(lambda: len(second.data) == 3)
    assert second.data == ["f1", "f2", "f3"]
    second.respond(200, lastEventId=3, closed=True, outcome="end")
    assert (await h.close(pub)).last_event_id == 3
    assert h.errors == []


async def test_a_server_that_holds_more_frames_than_were_written_is_another_stream(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    first = await h.request(0)
    await until(lambda: len(first.data) == 1)
    first.respond(408, lastEventId=7, closed=False, error="read_timeout")
    assert (await h.close(pub)).error == "gap"
    assert h.transport.closes == ["fail"]
    assert len(h.errors) == 1


async def test_min_open_holds_back_the_close_line_and_only_the_close_line(setup: Setup) -> None:
    h = setup(min_open=0.8)
    started = time.monotonic()
    pub = h.publisher()
    pub.send("quick")
    pub.send("answer")
    closing = asyncio.ensure_future(h.close(pub))
    request = await h.request(0)
    await until(lambda: len(request.data) == 2)
    assert time.monotonic() - started < 0.3, "frames are never delayed"
    await asyncio.sleep(0.2)
    assert request.closes == [], "the close line waits"
    await until(lambda: request.ended)
    assert time.monotonic() - started >= 0.75
    assert request.lines == ['{"data":"quick"}', '{"data":"answer"}', '{"close":"end"}']
    request.respond(200, lastEventId=2, closed=True, outcome="end")
    await closing


async def test_a_publisher_older_than_min_open_closes_at_once(setup: Setup) -> None:
    h = setup(min_open=0.2)
    pub = h.publisher()
    request = await h.request(0)
    await asyncio.sleep(0.25)
    closing = asyncio.ensure_future(h.close(pub))
    await until(lambda: request.ended, "the close line", 0.3)
    request.respond(200, lastEventId=0, closed=True, outcome="end")
    await closing


async def test_the_first_close_wins_and_every_call_returns_the_same_result(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    failing = asyncio.ensure_future(h.close(pub, "fail"))
    await asyncio.sleep(0.01)
    ending = asyncio.ensure_future(h.close(pub, "end"))
    request = await h.request(0)
    await until(lambda: request.ended)
    assert request.closes == ["fail"]
    request.respond(200, lastEventId=1, closed=True, outcome="fail", reason="publisher")
    first = await failing
    assert first.outcome == "fail"
    assert await ending is first
    assert await h.close(pub, "end") is first
    assert await h.close(pub) is first
    assert len(h.transport.requests) == 1
    assert h.transport.posts == []
    with pytest.raises(ValueError):
        await h.close(pub, "done")
    with pytest.raises(ValueError):
        await h.close(pub, "end", timeout=float("nan"))


async def test_fail_open_after_a_fatal_answer_send_returns_false_and_the_handler_ran_once(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    assert pub.send("a") is True
    (await h.request(0)).respond(401, error="unauthorized")
    await until(lambda: not pub.alive)
    assert pub.send("b") is False
    assert pub.send("c", "token") is False
    assert pub.frames_sent == 1, "only what was queued is counted"
    assert await h.close(pub, "end") == result(401, closed=False, error="unauthorized")
    assert await h.close(pub, "fail") == result(401, closed=False, error="unauthorized")
    assert len(h.errors) == 1
    assert pub.error is h.errors[0]


async def test_fail_open_without_a_handler_logs_one_warning_with_nothing_sensitive(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    h = Harness(kind, error_handler=None)
    pub = h.publisher()
    pub.send("private-data")
    request = await h.request(0)
    with caplog.at_level(logging.WARNING, logger="iq_stream"):
        request.respond(403, error="forbidden")
        await h.close(pub)
    records = [record for record in caplog.records if record.name == "iq_stream"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "403" in message
    for secret in ("private-data", SECRET, request.token):
        assert secret not in message


async def test_a_handler_that_raises_does_not_break_the_run(kind: str) -> None:
    def handler(error: IQError) -> None:
        raise RuntimeError("handler failed")

    h = Harness(kind, error_handler=handler)
    pub = h.publisher()
    (await h.request(0)).respond(401, error="unauthorized")
    assert (await h.close(pub)).status == 401
    assert pub.send("a") is False


async def test_on_error_raise_makes_send_and_close_raise_the_error(setup: Setup) -> None:
    h = setup(on_error="raise")
    pub = h.publisher()
    assert pub.send("a") is True
    (await h.request(0)).respond(401, error="unauthorized")
    await until(lambda: not pub.alive)
    with pytest.raises(IQError) as raised:
        pub.send("b")
    assert raised.value is pub.error
    assert raised.value.status == 401
    with pytest.raises(IQError) as closing:
        await h.close(pub)
    assert closing.value is pub.error
    with pytest.raises(IQError):
        await h.close(pub, "fail")
    assert len(h.errors) == 1, "the handler still runs once"


async def test_on_error_raise_does_not_turn_a_clean_run_into_an_error(setup: Setup) -> None:
    h = setup(on_error="raise")
    pub = h.publisher()
    pub.send("a")
    closing = asyncio.ensure_future(h.close(pub))
    request = await h.request(0)
    await until(lambda: request.ended)
    request.respond(200, lastEventId=1, closed=True, outcome="end")
    assert (await closing).closed is True
    assert pub.send("late") is False


async def test_the_queue_is_bounded_by_bytes_and_a_server_that_stops_reading_fails_the_stream(setup: Setup) -> None:
    h = setup(queue_bytes=1000)
    pub = h.publisher()
    request = await h.request(0)
    request.stalled = True
    h.transport.on_post = lambda body, n: closed_fail(0)
    data = "x" * 289  # each line is 301 bytes
    assert pub.send(data) is True
    await until(lambda: len(request.data) == 1)  # in flight, and never acknowledged
    assert pub.send(data) is True
    assert pub.send(data) is True
    assert pub.alive is True
    assert pub.send(data) is False, "a fourth line waiting for the server does not fit in 1000 bytes"
    assert pub.alive is False
    assert pub.error.code == "queue_full"
    await until(lambda: request.destroyed, "the stalled connection to be dropped")
    assert await h.close(pub) == result(
        0, closed=True, error="queue_full", last_event_id=0, outcome="fail", reason="publisher"
    )
    assert h.transport.closes == ["fail"]
    assert len(h.errors) == 1
    assert pub.frames_sent == 3


async def test_the_queue_bound_with_on_error_raise_raises_from_send(setup: Setup) -> None:
    h = setup(queue_bytes=100, on_error="raise")
    pub = h.publisher()
    (await h.request(0)).stalled = True
    pub.send("x" * 50)
    await asyncio.sleep(0.02)
    with pytest.raises(IQError) as raised:
        pub.send("x" * 50)
    assert raised.value.code == "queue_full"
    with pytest.raises(IQError):
        await h.close(pub)


async def test_mistakes_the_caller_can_fix_are_raised_at_once_even_from_a_publisher_that_gave_up(
    setup: Setup,
) -> None:
    h = setup(max_frame_bytes=100)
    pub = h.publisher()
    with pytest.raises(ValueError):
        pub.send(42)
    with pytest.raises(ValueError):
        pub.send("a", "iq:end")
    with pytest.raises(FrameTooLarge):
        pub.send("x" * 200)
    (await h.request(0)).respond(401)
    await h.close(pub)
    with pytest.raises(FrameTooLarge):
        pub.send("x" * 200)
    with pytest.raises(ValueError):
        pub.send("a", "bad name")
    assert pub.frames_sent == 0


async def test_send_accepts_a_line_of_exactly_the_limit_and_refuses_one_byte_more(setup: Setup) -> None:
    h = setup(max_frame_bytes=1024)
    pub = h.publisher()
    fits = "x" * (1024 - 11)
    assert pub.max_frame_bytes == 1024
    assert pub.frame_size(fits) == 1024
    assert pub.send(fits) is True
    assert pub.send("é" * ((1024 - 11) // 2) + "x") is True, "bytes are counted, not characters"
    with pytest.raises(FrameTooLarge) as raised:
        pub.send(fits + "x")
    assert isinstance(raised.value, IQError)
    assert (raised.value.size, raised.value.limit) == (1025, 1024)
    assert "xxx" not in str(raised.value), "no frame data in the message"
    with pytest.raises(FrameTooLarge):
        pub.send("é" * 507)
    with pytest.raises(FrameTooLarge):
        pub.send("x" * (1024 - 26), "token")
    assert pub.alive is True, "a refused frame is the caller's to fix: the publisher goes on"
    assert pub.send("after") is True
    assert pub.frames_sent == 3
    request = await h.request(0)
    await until(lambda: len(request.data) == 3)
    request.respond(200, lastEventId=3, closed=True, outcome="end")
    await h.close(pub)
    assert [len(data) for data in request.data] == [1013, 507, 5]
    assert h.errors == []


async def test_silence_is_answered_with_keepalive_lines_which_are_not_frames(setup: Setup) -> None:
    h = setup(keepalive=0.05)
    pub = h.publisher()
    pub.send("a")
    request = await h.request(0)
    await until(lambda: request.keepalives >= 3)
    assert request.data == ["a"]
    assert pub.frames_sent == 1
    pub.send("b")
    await until(lambda: len(request.data) == 2)
    # A reopened request resends by frame count: keepalives must not have been counted.
    request.respond(408, lastEventId=1, closed=False, error="read_timeout")
    second = await h.request(1)
    await until(lambda: len(second.data) == 1)
    assert second.data == ["b"]
    second.respond(200, lastEventId=2, closed=True, outcome="end")
    await h.close(pub)


async def test_keepalive_none_writes_no_empty_line(setup: Setup) -> None:
    h = setup(keepalive=None)
    pub = h.publisher()
    pub.send("a")
    request = await h.request(0)
    await asyncio.sleep(0.15)
    assert request.lines == ['{"data":"a"}']
    request.respond(200, lastEventId=1, closed=True, outcome="end")
    await h.close(pub)


async def test_a_write_restarts_the_keepalive_clock(setup: Setup) -> None:
    h = setup(keepalive=0.4)
    pub = h.publisher()
    request = await h.request(0)
    for i in range(6):
        pub.send(f"f{i}")
        await asyncio.sleep(0.1)
    assert request.keepalives == 0
    request.respond(200, lastEventId=6, closed=True, outcome="end")
    await h.close(pub)


async def test_abort_drops_the_connection_without_closing_the_stream(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    request = await h.request(0)
    await until(lambda: len(request.data) == 1)
    pub.abort()
    await until(lambda: request.destroyed)
    assert pub.alive is False
    assert pub.send("b") is False
    assert await h.close(pub) == result(0, closed=False)
    pub.abort()
    await asyncio.sleep(0.03)
    assert request.closes == []
    assert h.transport.posts == [], "no probe, no close: the stream is another process's now"
    assert len(h.transport.requests) == 1
    assert pub.error is None
    assert h.errors == []


async def test_close_gives_up_after_its_timeout_and_says_the_stream_failed(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    pub.send("a")
    h.transport.on_post = lambda body, n: closed_fail(1)
    started = time.monotonic()
    outcome = await h.close(pub, "end", timeout=0.3)
    assert 0.25 <= time.monotonic() - started < 1.5
    assert outcome == result(0, closed=True, error="close_timeout", last_event_id=1, outcome="fail", reason="publisher")
    assert h.transport.requests[0].destroyed is True
    assert h.transport.closes == ["fail"]
    assert h.errors[0].code == "close_timeout"


async def test_the_queue_bound_counts_what_waits_during_a_recovery_and_lets_go_of_what_was_written(
    setup: Setup,
) -> None:
    h = setup(queue_bytes=1000)
    pub = h.publisher()
    request = await h.request(0)
    data = "x" * 289
    for i in range(10):
        assert pub.send(data) is True, "a healthy request empties the queue"
        await until(lambda i=i: len(request.data) == i + 1)
    probe = Pending()
    h.transport.on_post = lambda body, n: probe if body == "" else closed_fail(10)
    request.respond(0)
    await until(lambda: h.transport.probes == 1)
    assert [pub.send(data) for _ in range(4)] == [True, True, True, False]
    assert h.errors[0].code == "queue_full"
    probe.resolve(open_(10))
    assert (await h.close(pub)).error == "queue_full"
    assert len(h.transport.requests) == 1


async def test_frame_size_measures_without_sending(setup: Setup) -> None:
    h = setup()
    pub = h.publisher()
    assert pub.frame_size("Hi") == 13
    assert pub.frame_size("Hi", "token") == 29
    assert pub.frames_sent == 0
    assert pub.max_frame_bytes == 65536
    assert pub.stream_id == STREAM
    pub.abort()


async def test_retry_budget_zero_turns_retries_off_not_the_time_giving_up_may_take(setup: Setup) -> None:
    h = setup(retry_budget=0)
    pub = h.publisher()
    h.transport.on_post = lambda body, n: closed_fail(0)
    (await h.request(0)).respond(400, error="bad_request")
    assert (await h.close(pub)).outcome == "fail"
    assert len(h.transport.timeouts) == 1
    assert 2.9 < h.transport.timeouts[0] <= 3.0, "giving up has its own bound"


async def test_the_context_manager_closes_with_end_and_with_fail_when_the_block_raises(setup: Setup) -> None:
    h = setup()
    h.transport.on_open = lambda request, n: None

    async def run(raises: BaseException | None) -> object:
        pub = h.publisher()
        request = await h.request(len(h.transport.requests) - 1) if h.transport.requests else await h.request(0)

        async def answer_close() -> None:
            await until(lambda: request.ended)
            request.respond(200, lastEventId=1, closed=True, outcome=request.closes[0])

        answering = asyncio.ensure_future(answer_close())
        try:
            if h.kind == "sync":

                def block() -> None:
                    with pub:
                        pub.send("a")
                        if raises is not None:
                            raise raises

                await asyncio.to_thread(block)
            else:
                async with pub:
                    pub.send("a")
                    if raises is not None:
                        raise raises
        except BaseException as error:
            await answering
            return error
        await answering
        return request.closes

    assert await run(None) == ["end"]
    h.transport.requests.clear()
    error = await run(RuntimeError("the run failed"))
    assert isinstance(error, RuntimeError)
    assert h.transport.requests[0].closes == ["fail"]


async def test_the_context_manager_lets_the_blocks_own_exception_through_in_raise_mode(setup: Setup) -> None:
    h = setup(on_error="raise")
    pub = h.publisher()
    (await h.request(0)).respond(401, error="unauthorized")
    await until(lambda: not pub.alive)
    with pytest.raises(KeyError):
        if h.kind == "sync":
            with pub:
                raise KeyError("the run's own error")
        else:
            async with pub:
                raise KeyError("the run's own error")
    with pytest.raises(IQError):
        if h.kind == "sync":
            with pub:
                pass
        else:
            async with pub:
                pass
