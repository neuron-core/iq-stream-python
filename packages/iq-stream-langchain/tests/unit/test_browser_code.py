"""The browser code the README shows is the code the front-end contract tests run (tests/frontend/langgraph.mjs)."""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]


def test_the_readme_shows_the_tested_bridge() -> None:
    blocks = re.findall(r"```js\n(.*?)```", (PACKAGE / "README.md").read_text(encoding="utf-8"), re.DOTALL)
    assert blocks == [(PACKAGE / "tests" / "frontend" / "langgraph.mjs").read_text(encoding="utf-8")]
