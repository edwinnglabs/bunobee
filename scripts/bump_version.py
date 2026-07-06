#!/usr/bin/env python3
"""Bump bunobee's version, staying inside the ``0.0.x`` release lane.

Releases are intentionally pinned to the ``0.0.x`` patch line. Graduating to
``0.x`` / ``1.0`` is a deliberate, separate decision and is **not** automated
here, so this helper only ever moves the patch (or ``dev``) segment and refuses
to touch a version that has already left the lane. The publish workflow enforces
the same rule, so this script keeps the local edit honest.

Examples
--------
    python scripts/bump_version.py            # 0.0.4      -> 0.0.5
    python scripts/bump_version.py --dev      # 0.0.4      -> 0.0.5.dev0
    python scripts/bump_version.py --dev      # 0.0.5.dev0 -> 0.0.5.dev1
    python scripts/bump_version.py            # 0.0.5.dev1 -> 0.0.5   (finalize)
    python scripts/bump_version.py --show     # print the current version only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_VERSION_LINE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:\.?dev(\d+))?$")


def read_version(text: str) -> tuple[re.Match[str], str]:
    """Locate the ``version = "..."`` line and return its match and value.

    Parameters
    ----------
    text : str
        Full contents of ``pyproject.toml``.

    Returns
    -------
    tuple of (re.Match, str)
        The regex match (used to splice the replacement back in) and the
        current version string.
    """
    match = _VERSION_LINE.search(text)
    if match is None:
        sys.exit('error: no `version = "..."` line found in pyproject.toml')
    return match, match.group(2)


def next_version(version: str, *, dev: bool) -> str:
    """Compute the next in-lane version.

    Parameters
    ----------
    version : str
        The current version, e.g. ``"0.0.4"`` or ``"0.0.5.dev0"``.
    dev : bool
        When ``True``, bump/append a ``.devN`` pre-release segment; otherwise
        advance the patch (or finalize a dev build in place).

    Returns
    -------
    str
        The bumped version string.
    """
    parsed = _VERSION.match(version)
    if parsed is None:
        sys.exit(f"error: cannot parse version {version!r} (expected e.g. 0.0.4 or 0.0.4.dev0)")
    major, minor, patch, dev_num = (int(g) if g is not None else None for g in parsed.groups())
    if (major, minor) != (0, 0):
        sys.exit(
            f"error: {version} has left the 0.0.x lane; a 0.x/1.0 bump is a deliberate "
            "lift and is not automated by this script"
        )
    if dev:
        if dev_num is not None:
            return f"0.0.{patch}.dev{dev_num + 1}"
        return f"0.0.{patch + 1}.dev0"
    # Plain bump: finalize a dev build in place, else advance the patch.
    if dev_num is not None:
        return f"0.0.{patch}"
    return f"0.0.{patch + 1}"


def main() -> None:
    """Parse arguments and rewrite the version in ``pyproject.toml``."""
    parser = argparse.ArgumentParser(description="Bump bunobee's version within the 0.0.x lane.")
    parser.add_argument("--dev", action="store_true", help="bump/append a .devN pre-release segment")
    parser.add_argument("--show", action="store_true", help="print the current version and exit")
    args = parser.parse_args()

    text = PYPROJECT.read_text()
    match, current = read_version(text)

    if args.show:
        print(current)
        return

    new = next_version(current, dev=args.dev)
    PYPROJECT.write_text(text[: match.start(2)] + new + text[match.end(2) :])
    print(f"{current} -> {new}")


if __name__ == "__main__":
    main()
