"""The repository around the package: scripts/servers.py starts the short server with its own copy of
SHORT_SETTINGS, since it runs before any package is installed, and the two must not drift apart; and the test suite
is laid out so that pytest collects all of it in one run."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from iq_stream.testing import SHORT_SETTINGS

PACKAGE = Path(__file__).resolve().parents[2]
SCRIPT = PACKAGE.parents[1] / "scripts" / "servers.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="not in a checkout of the repository")
def test_scripts_servers_py_starts_the_short_server_with_the_kits_short_settings() -> None:
    spec = importlib.util.spec_from_file_location("servers_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert dict(SHORT_SETTINGS) == script.SHORT_SETTINGS


def test_the_whole_suite_collects_in_one_run() -> None:
    # pytest refuses pytest_plugins in a conftest it does not load at startup: `pytest tests` and an IDE's run of
    # the whole suite would stop at collection.
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=PACKAGE,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "tests/integration/test_scenarios.py" in process.stdout


MATRIX = PACKAGE.parents[1] / "scripts" / "ci_matrix.py"


@pytest.mark.skipif(not MATRIX.exists(), reason="not in a checkout of the repository")
@pytest.mark.skipif(sys.version_info < (3, 11), reason="the script reads TOML with tomllib, as CI runs it on 3.12")
def test_ci_runs_every_package_at_the_floor_it_declares() -> None:
    process = subprocess.run([sys.executable, str(MATRIX)], capture_output=True, text=True, timeout=60, check=True)
    outputs = dict(line.split("=", 1) for line in process.stdout.splitlines())
    floor = json.loads(outputs["floor"])["include"]
    assert {"package": "iq-stream", "python": "3.10", "pins": "httpx==0.27.2"} in floor
    unit = json.loads(outputs["unit"])["include"]
    assert {entry["package"] for entry in floor} <= {entry["package"] for entry in unit}
