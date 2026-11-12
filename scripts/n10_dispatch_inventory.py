#!/usr/bin/env python3
"""Inventory of the field-or-array dispatch that the N10 retyping moves (ROM-85, lots 1a and 1b, R3).

Every comparison in the core Python (``python/pops``) whose one side reads the spelling of a value
type and whose other side holds a field or finite spelling is one row: file, line, kind and the
text of the line where the comparison starts. The file is read as a syntax tree, so a comparison
is found whatever its layout:

* the spelling is read as ``x.vtype``, ``x["vtype"]``, ``x.get("vtype")`` or
  ``getattr(x, "vtype", ...)``, or through a name bound to one of these in the same module
  (``kind = sig["vtype"]``, then ``kind in (...)``);
* the other side is a string, a tuple, list or set of strings (on any number of lines), a
  ``frozenset``/``set``/``tuple`` of them, or a module constant holding them
  (``value.vtype in _DISCRETE``).

The kind is read from the spellings compared:

* ``field``: ``scalar_field``, or both ``state`` and ``rhs``: the comparison asks « is this a
  field »;
* ``role``: exactly one of ``state`` and ``rhs``: it tells a state from its rate, which stays a
  role test after the retyping;
* ``finite``: ``array``, ``index_set`` or ``index``: the other side of the same dispatch.

Lot 1a listed the lines with a line pattern, which missed the bound names, the module constants
and the tuples written over several lines (review of #777, B1). Lot 1b moved every ``field`` and
``finite`` comparison onto the one function of ``pops._ir.data_model`` (``value_kind``,
``is_field_leaf``, ``is_finite_array``, ``is_finite_index``, ``field_role``), which reads the kind
from the head of the IR type. Only that module still compares those spellings, and the architecture
test of N10 holds it. Writing a signature (``vtype="array"``) is its encoding, not a comparison,
and is not a row. ``--write`` regenerates the CSV, ``--check`` reports whether it is current.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "design" / "n10_dispatch_inventory.csv"
FIELD = frozenset(("scalar_field",))
ROLES = frozenset(("state", "rhs"))
FINITE = frozenset(("array", "index_set", "index"))


def _reads_spelling(node: ast.AST) -> bool:
    """Whether ``node`` reads the spelling of a value type."""
    if isinstance(node, ast.NamedExpr):
        return _reads_spelling(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr == "vtype"
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "vtype"
    if isinstance(node, ast.Call):
        args = node.args
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            return bool(args) and isinstance(args[0], ast.Constant) and args[0].value == "vtype"
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            return (len(args) >= 2 and isinstance(args[1], ast.Constant)
                    and args[1].value == "vtype")
    return False


def _strings(node: ast.AST, constants: dict[str, frozenset[str]]) -> frozenset[str]:
    """The strings that ``node`` spells: a literal, a collection of them, or a module constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset((node.value,))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return frozenset().union(*(_strings(element, constants) for element in node.elts))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set", "tuple") and len(node.args) == 1):
        return _strings(node.args[0], constants)
    if isinstance(node, ast.Name):
        return constants.get(node.id, frozenset())
    return frozenset()


def _classify(spellings: frozenset[str]) -> str | None:
    if spellings & FIELD or ROLES <= spellings:
        return "field"
    if spellings & ROLES:
        return "role"
    if spellings & FINITE:
        return "finite"
    return None


def rows_of(source: str) -> list[tuple[int, str, str]]:
    """``(line, kind, text)`` of every comparison of a field, role or finite spelling in ``source``."""
    tree = ast.parse(source)
    lines = source.splitlines()
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            spelled = _strings(node.value, {})
            if spelled:
                constants[node.targets[0].id] = spelled
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _reads_spelling(node.value):
            bound.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None \
                and _reads_spelling(node.value) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = (node.left, *node.comparators)
        reads = [operand for operand in operands if _reads_spelling(operand)
                 or (isinstance(operand, ast.Name) and operand.id in bound)]
        if not reads:
            continue
        spelled = frozenset().union(*(_strings(operand, constants) for operand in operands
                                      if not any(operand is read for read in reads)))
        kind = _classify(spelled)
        if kind is not None:
            found.append((node.lineno, kind, lines[node.lineno - 1].strip()))
    return sorted(found)


def rows() -> list[tuple[str, int, str, str]]:
    found = []
    for path in sorted((ROOT / "python" / "pops").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        for number, kind, text in rows_of(path.read_text(encoding="utf-8")):
            found.append((relative, number, kind, text))
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
