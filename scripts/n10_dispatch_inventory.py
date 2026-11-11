#!/usr/bin/env python3
"""Inventory of the field-or-array dispatch that the N10 retyping moves (ROM-85, lot 1a, R3).

Every line of the core Python (``python/pops``) that reads a value type of the Program builder to
decide between a field leaf, a role or a finite value is one row: file, line, kind and the text.
The kind is read from the spellings the line compares:

* ``field``: a field spelling (``scalar_field``, ``_FIELD``, ``is_field``) or both ``state`` and
  ``rhs``: the line asks « is this a field »;
* ``role``: exactly one of ``state`` and ``rhs``: the line tells a state from its rate, which
  stays a role test after the retyping;
* ``finite``: ``array`` or ``index_set``: the other side of the same dispatch.

Lot 1b moves the ``field`` and ``finite`` rows onto ``ir_type`` and the plan's layout and closes
them row by row; ``role`` rows stay. The kind is a reading of the text, which the review of each
row in 1b confirms. ``--write`` regenerates the CSV, ``--check`` reports whether it is current.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "design" / "n10_dispatch_inventory.csv"
_READS_TYPE = re.compile(r"vtype|_FIELD\b|is_field")
_FIELD = re.compile(r"\"scalar_field\"|'scalar_field'|_FIELD\b|is_field")
_STATE = re.compile(r"[\"']state[\"']")
_RHS = re.compile(r"[\"']rhs[\"']")
_FINITE = re.compile(r"[\"'](array|index_set)[\"']")


def _kind(text: str) -> str | None:
    if not _READS_TYPE.search(text):
        return None
    if _FIELD.search(text) or (_STATE.search(text) and _RHS.search(text)):
        return "field"
    if _STATE.search(text) or _RHS.search(text):
        return "role"
    if _FINITE.search(text):
        return "finite"
    return None


def rows() -> list[tuple[str, int, str, str]]:
    found = []
    for path in sorted((ROOT / "python" / "pops").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            kind = _kind(line)
            if kind is not None:
                found.append((relative, number, kind, line.strip()))
    return found


def render() -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("file", "line", "kind", "text"))
    writer.writerows(rows())
    return stream.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.write:
        TARGET.write_text(text, encoding="utf-8")
        return 0
    current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
    if current != text:
        print("docs/design/n10_dispatch_inventory.csv is stale; run scripts/n10_dispatch_inventory.py --write")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
