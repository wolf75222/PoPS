#!/usr/bin/env python3
"""Inventory of the field-or-array dispatch that the N10 retyping moves (ROM-85, lots 1a and 1b, R3).

Every line of the core Python (``python/pops``) that compares the spelling of a value type
(``vtype``, as an attribute, a key or ``.get``) with a field or finite spelling is one row: file,
line, kind and the text. The kind is read from the spellings compared:

* ``field``: ``scalar_field``, or both ``state`` and ``rhs``: the line asks « is this a field »;
* ``role``: exactly one of ``state`` and ``rhs``: the line tells a state from its rate, which
  stays a role test after the retyping;
* ``finite``: ``array``, ``index_set`` or ``index``: the other side of the same dispatch.

Lot 1a listed the lines; lot 1b moved the ``field`` and ``finite`` rows onto the one function of
``pops._ir.data_model`` (``value_kind``, ``is_field_leaf``, ``is_finite_array``,
``is_finite_index``, ``field_role``), which reads the kind from the head of the IR type. Only that
module still compares those spellings, and the architecture test of N10 holds it. Writing a
signature (``vtype="array"``) is its encoding, not a test, and is not a row. ``--write``
regenerates the CSV, ``--check`` reports whether it is current.
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
_ACCESS = r"""(?:\.vtype|\[["']vtype["']\]|\.get\(["']vtype["']\))"""
_SPELLINGS = r"""["'](?:state|rhs|scalar_field|array|index_set|index)["']"""
# The spelling is the compared operand: ``<access> == "x"``, ``<access> in (..., "x", ...)``, or
# ``"x" == <access>``.
_COMPARES = re.compile(
    _ACCESS + r"\s*(?:==|!=)\s*" + _SPELLINGS
    + r"|" + _ACCESS + r"""\s+(?:not\s+)?in\s+[\(\{\[][^\)\}\]]*""" + _SPELLINGS
    + r"|" + _SPELLINGS + r"\s*(?:==|!=)\s*[\w\.\[\]\"']*" + _ACCESS)
_FIELD = re.compile(r"""["']scalar_field["']""")
_STATE = re.compile(r"[\"']state[\"']")
_RHS = re.compile(r"[\"']rhs[\"']")
_FINITE = re.compile(r"[\"'](array|index_set|index)[\"']")


def _kind(text: str) -> str | None:
    if not _COMPARES.search(text):
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
