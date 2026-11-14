from __future__ import annotations

import json

import pytest

from iq_stream import FrameTooLarge, IQError
from iq_stream._frame import checked_frame, close_line, encode_frame, frame_size


@pytest.mark.parametrize(
    ("data", "event", "line"),
    [
        ("Hi", None, '{"data":"Hi"}\n'),
        ("Hi", "token", '{"event":"token","data":"Hi"}\n'),
        ("", None, '{"data":""}\n'),
        ("héllo wörld", None, '{"data":"héllo wörld"}\n'),
        ("日本語 🎉", None, '{"data":"日本語 🎉"}\n'),
        ("a\nb\r\nc\td", None, '{"data":"a\\nb\\r\\nc\\td"}\n'),
        ('say "hi" \\ now', None, '{"data":"say \\"hi\\" \\\\ now"}\n'),
        ("https://example.com/a/b", None, '{"data":"https://example.com/a/b"}\n'),
        ('{"type":"TEXT","delta":"é"}', None, '{"data":"{\\"type\\":\\"TEXT\\",\\"delta\\":\\"é\\"}"}\n'),
        ("\u0000\u001f\u007f", None, '{"data":"\\u0000\\u001f\u007f"}\n'),
        ("\u2028", None, '{"data":"\u2028"}\n'),
        ("half an emoji \ud83d", None, '{"data":"half an emoji \\ud83d"}\n'),
    ],
)
def test_a_frame_is_one_compact_json_line_byte_for_byte_as_the_javascript_core_writes_it(
    data: str, event: str | None, line: str
) -> None:
    encoded = encode_frame(data, event)
    assert encoded == line.encode("utf-8")
    assert encoded.decode("utf-8") == line
    if "\ud83d" not in data:
        assert json.loads(line) == ({"data": data} if event is None else {"event": event, "data": data})


def test_the_size_of_a_frame_is_its_line_in_utf8_bytes_without_the_newline() -> None:
    assert frame_size("Hi") == 13
    assert frame_size("é") == 11 + 2
    assert frame_size("🎉") == 11 + 4
    assert frame_size("a\nb") == 11 + 4
    assert frame_size('"') == 11 + 2
    assert frame_size("x", "token") == len('{"event":"token","data":"x"}')


@pytest.mark.parametrize("event", ["a", "A", "0", "_", ".", "-", "run.step-1_done", "x" * 64, "end", "fail", "gone"])
def test_event_names_at_each_boundary_of_the_pattern_are_accepted(event: str) -> None:
    assert json.loads(encode_frame("d", event)) == {"event": event, "data": "d"}


@pytest.mark.parametrize("event", ["", "x" * 65, "iq:end", "a:b", "a b", "a/b", "é", "a\n", "٣", 42, {}])
def test_event_names_outside_the_pattern_are_refused(event: object) -> None:
    with pytest.raises(ValueError):
        encode_frame("d", event)  # type: ignore[arg-type]


@pytest.mark.parametrize("data", [None, 42, b"bytes", {"a": 1}, ["a"]])
def test_data_must_be_a_string(data: object) -> None:
    with pytest.raises(ValueError):
        encode_frame(data)  # type: ignore[arg-type]


def test_close_lines() -> None:
    assert close_line("end") == b'{"close":"end"}\n'
    assert close_line("fail") == b'{"close":"fail"}\n'


def test_a_line_of_exactly_the_limit_fits_and_one_byte_more_does_not() -> None:
    fits = "x" * (1024 - 11)
    assert checked_frame(fits, None, 1024) == b'{"data":"' + fits.encode() + b'"}\n'
    with pytest.raises(FrameTooLarge) as raised:
        checked_frame(fits + "x", None, 1024)
    assert isinstance(raised.value, IQError)
    assert (raised.value.size, raised.value.limit) == (1025, 1024)
    assert str(raised.value) == "The line is 1025 bytes; the limit is 1024."
    fits_named = "x" * (1024 - 27)
    assert len(checked_frame(fits_named, "token", 1024)) == 1025
    with pytest.raises(FrameTooLarge):
        checked_frame(fits_named + "x", "token", 1024)
    with pytest.raises(FrameTooLarge):
        checked_frame("é" * 507, None, 1024)
