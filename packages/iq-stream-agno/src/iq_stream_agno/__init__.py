"""Publish Agno runs (agents, teams, workflows) to browsers through an IQ-Stream server.

``publish()`` and ``apublish()`` wrap the iterator of ``run(stream=True)`` / ``arun(stream=True)``: every event goes on
to the caller unchanged and becomes a frame, and the stream closes when the run ends. ``start_run()`` starts a run in
a task of its own and returns its ticket at once. The AgentOS interface is ``iq_stream_agno.os.IQStream``.
"""

from ._common import MASKED_ERROR, Protocol
from ._runner import PublishOptions, StreamHandle, continue_run, running, start_run
from ._tee import RunStatus, RunSummary, apublish, publish

__version__ = "0.1.0"

__all__ = [
    "MASKED_ERROR",
    "Protocol",
    "PublishOptions",
    "RunStatus",
    "RunSummary",
    "StreamHandle",
    "__version__",
    "apublish",
    "continue_run",
    "publish",
    "running",
    "start_run",
]
