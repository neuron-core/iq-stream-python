"""The Python blocks of the README, and what the README says is the application's."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
BLOCKS: list[str] = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.DOTALL)
ROUTE = next(block for block in BLOCKS if "FastAPI()" in block)
SCRIPTS = [block for block in BLOCKS if block is not ROUTE]

PRELUDE = f"""
import sys
sys.path.insert(0, {str(TESTS)!r})
from crew_fakes import make_crew

crew = make_crew()
saved = {{}}


def build_crew():
    return make_crew()


def save_answer(stream_id, text):
    saved[stream_id] = text


async def asave_answer(stream_id, text):
    saved[stream_id] = text

"""
