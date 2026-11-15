"""What a run reports about frames it could not send as they were: logged once per run, or given to ``on_oversize``.
A log line names the stream, the part's type and sizes, never frame data."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("iq_stream_langchain")


@dataclass(frozen=True)
class Oversize:
    """A part whose frame was over the publisher's ``max_frame_bytes``."""

    #: The part's ``type``: ``messages``, ``updates``, ...
    part_type: str
    #: Bytes of the frame that did not fit.
    size: int
    limit: int
    #: True when it was sent cut down to fit; False when it was not sent.
    truncated: bool


class Reports:
    """One run's reports. ``unit`` is what a frame carries, for the log lines: a ``part`` or an ``event``."""

    def __init__(self, stream_id: str, on_oversize: Callable[[Oversize], None] | None, unit: str) -> None:
        self.stream_id = stream_id
        self.on_oversize = on_oversize
        self.unit = unit
        self.unencodable_reported = False
        self.oversize_reported = False
        self.hook_failure_reported = False

    def unencodable(self, kind: str) -> None:
        """One frame is lost, never the run."""
        if not self.unencodable_reported:
            self.unencodable_reported = True
            logger.warning(
                "IQ stream %s: a %s %s could not be encoded and was not sent; later ones are not reported.",
                self.stream_id,
                kind,
                self.unit,
            )

    def oversize(self, report: Oversize) -> None:
        if self.on_oversize is None:
            if not self.oversize_reported:
                self.oversize_reported = True
                logger.warning(
                    "IQ stream %s: a %s %s of %d bytes is over the frame limit of %d bytes and was %s; "
                    "later ones are not reported.",
                    self.stream_id,
                    report.part_type,
                    self.unit,
                    report.size,
                    report.limit,
                    "truncated" if report.truncated else "not sent",
                )
            return
        try:
            self.on_oversize(report)
        except Exception as error:
            # A hook that reports on the run must not fail it.
            if not self.hook_failure_reported:
                self.hook_failure_reported = True
                logger.warning(
                    "IQ stream %s: on_oversize raised %s; later failures are not reported.",
                    self.stream_id,
                    type(error).__name__,
                )
