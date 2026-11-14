#!/usr/bin/env python3
"""Prints the CI matrices as GitHub Actions outputs, from the packages in packages/*/pyproject.toml.

Each package is tested on the Python versions its classifiers name ("Programming Language :: Python :: 3.X"):
unit tests on all of them, integration tests on the lowest and the highest. A new package needs no change to CI.

    python3 scripts/ci_matrix.py >> "$GITHUB_OUTPUT"
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_VERSION = re.compile(r"Programming Language :: Python :: (3\.\d+)$")


def versions(pyproject: Path) -> list[str]:
    classifiers = tomllib.loads(pyproject.read_text())["project"].get("classifiers", [])
    found = {match.group(1) for classifier in classifiers if (match := _VERSION.match(classifier))}
    return sorted(found, key=lambda version: tuple(int(part) for part in version.split(".")))


def main() -> None:
    unit: list[dict[str, str]] = []
    integration: list[dict[str, str]] = []
    for pyproject in sorted(ROOT.glob("packages/*/pyproject.toml")):
        package = pyproject.parent.name
        supported = versions(pyproject)
        if not supported:
            raise SystemExit(f"{pyproject.relative_to(ROOT)} names no 'Programming Language :: Python :: 3.X'.")
        unit += [{"package": package, "python": version} for version in supported]
        integration += [
            {"package": package, "python": version} for version in dict.fromkeys([supported[0], supported[-1]])
        ]
    print(f"unit={json.dumps({'include': unit})}")
    print(f"integration={json.dumps({'include': integration})}")


if __name__ == "__main__":
    main()
