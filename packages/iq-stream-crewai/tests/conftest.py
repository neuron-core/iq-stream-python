import os
import shutil
import tempfile
from collections.abc import Iterator

import pytest

# Before CrewAI is imported: no telemetry or tracing, and the database and settings CrewAI writes under the user
# data directory on import and on every kickoff go to a directory of the session's own.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
_DATA_HOME = tempfile.mkdtemp(prefix="iq-stream-crewai-tests-")
os.environ["XDG_DATA_HOME"] = _DATA_HOME

pytest_plugins = ["iq_stream.testing.pytest_plugin"]


@pytest.fixture(scope="session", autouse=True)
def _crewai_data_home() -> Iterator[None]:
    yield
    shutil.rmtree(_DATA_HOME, ignore_errors=True)
