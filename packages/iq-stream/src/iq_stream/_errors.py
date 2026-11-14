from __future__ import annotations


class IQError(Exception):
    """Base class of every error this package raises.

    It carries the stream id, the HTTP status and the server's error code when there are any, and never a
    token, the app secret or frame data.
    """

    #: The stream the error concerns.
    stream_id: str | None
    #: The HTTP status of the server's answer, when there was one.
    status: int | None
    #: The server's error code (``stream_closed``, ``frame_too_large``, ...) or the publisher's own
    #: (``gap``, ``queue_full``, ``close_timeout``, ``unreachable``, ``cancelled``).
    code: str | None

    def __init__(
        self, message: str, *, stream_id: str | None = None, status: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.stream_id = stream_id
        self.status = status
        self.code = code

    def __reduce__(self) -> tuple[object, ...]:
        return (_rebuild, (type(self), str(self), self.__dict__.copy()))


class FrameTooLarge(IQError):
    """Raised by ``send()`` for a line over the limit, so that the caller can truncate or split the payload."""

    #: Bytes of the line, envelope and escapes included, without its newline.
    size: int
    #: The limit it had to fit.
    limit: int

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"The line is {size} bytes; the limit is {limit}.")
        self.size = size
        self.limit = limit


def _rebuild(cls: type[IQError], message: str, state: dict[str, object]) -> IQError:
    error = Exception.__new__(cls)
    Exception.__init__(error, message)
    error.__dict__.update(state)
    return error
