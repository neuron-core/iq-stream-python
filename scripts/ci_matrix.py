#!/usr/bin/env python3
"""Prints the CI matrices as GitHub Actions outputs, from the packages in packages/*/pyproject.toml.

Each package is tested on the Python versions its classifiers name ("Programming Language :: Python :: 3.X"):
unit tests on all of them, integration tests on the lowest and the highest. A package that declares the lowest
versions of its dependencies it supports, as

    [tool.iq-stream]
    floor = ["httpx==0.27.2"]

also runs its unit tests on its lowest Python with exactly those versions installed. A new package needs no change
to CI.

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


def floor_of(pyproject: Path) -> list[str]:
    pins = tomllib.loads(pyproject.read_text()).get("tool", {}).get("iq-stream", {}).get("floor", [])
    if not isinstance(pins, list) or not all(isinstance(pin, str) and "==" in pin for pin in pins):
        raise SystemExit(f'{pyproject.relative_to(ROOT)}: [tool.iq-stream] floor is a list of "name==version".')
    return pins


def main() -> None:
    unit: list[dict[str, str]] = []
    integration: list[dict[str, str]] = []
    floor: list[dict[str, str]] = []
    for pyproject in sorted(ROOT.glob("packages/*/pyproject.toml")):
        package = pyproject.parent.name
        supported = versions(pyproject)
        if not supported:
            raise SystemExit(f"{pyproject.relative_to(ROOT)} names no 'Programming Language :: Python :: 3.X'.")
        unit += [{"package": package, "python": version} for version in supported]
        pins = floor_of(pyproject)
        if pins:
            floor.append({"package": package, "python": supported[0], "pins": " ".join(pins)})
        integration += [
            {"package": package, "python": version} for version in dict.fromkeys([supported[0], supported[-1]])
        ]
    print(f"unit={json.dumps({'include': unit})}")
    print(f"integration={json.dumps({'include': integration})}")
    print(f"floor={json.dumps({'include': floor})}")


if __name__ == "__main__":
    main()
