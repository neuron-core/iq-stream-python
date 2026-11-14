"""The Python client of the IQ-Stream server: publish an agent's answer to browsers over Server-Sent Events."""

from ._answer import PublishResult
from ._apublisher import AsyncPublisher
from ._client import IQClient
from ._errors import FrameTooLarge, IQError
from ._interfaces import AsyncPublisherLike, PublisherLike
from ._protocol import DEFAULT_MAX_FRAME_BYTES, Outcome, Scope, Ticket
from ._publisher import Publisher
from ._token import new_stream_id
from ._transport import USER_AGENT

__version__ = USER_AGENT.rpartition("/")[2]

__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "AsyncPublisher",
    "AsyncPublisherLike",
    "FrameTooLarge",
    "IQClient",
    "IQError",
    "Outcome",
    "PublishResult",
    "Publisher",
    "PublisherLike",
    "Scope",
    "Ticket",
    "__version__",
    "new_stream_id",
]
