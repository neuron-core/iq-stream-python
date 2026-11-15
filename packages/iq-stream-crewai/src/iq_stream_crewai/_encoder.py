from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

from crewai.events.base_events import BaseEvent

from ._limits import Frame

S = TypeVar("S")


class Encoder(Protocol[S]):
    """Turns CrewAI events into frames, for ``protocol=`` when neither built-in protocol fits.

    ``snapshot`` runs in the thread that emitted the event, with no lock held: it copies what ``frames`` needs, or
    returns ``None`` to skip the event. The other methods run under the stream's lock, one call at a time, so the
    encoder may keep state. Frames are ``(event name or None, data)``; a frame over the publisher's
    ``max_frame_bytes`` is dropped and recorded in ``CrewAIStream.errors``.
    """

    def snapshot(self, source: object, event: BaseEvent) -> S | None: ...

    def start(self) -> Iterable[Frame]:
        """The frames sent when the block is entered."""
        ...

    def frames(self, snapshot: S) -> Iterable[Frame]:
        """The frames of one event."""
        ...

    def finish(self, error: BaseException | None) -> Iterable[Frame]:
        """The frames sent when the block exits, with the exception it exits with."""
        ...
