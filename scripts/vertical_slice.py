#!/usr/bin/env python3
"""The registry of the vertical slice of the exit of phase 1 (OD31; spec 0.3, MG6).

``tests/gates/vertical_slice.toml`` lists, in the order in which the case file of the slice meets
them, the constructs that PoPS does not run yet (``[[piece]]``): the statement of the file, the
refusal that it raises and the owner that lifts it. The test of the slice
(``tests/python/integration/surface/test_vertical_slice_t02.py``) runs the file and checks that it
stops on the first piece. The list only shrinks (MG6) and is empty at the exit of phase 1.

``python3 scripts/vertical_slice.py check [--check-base REVISION]`` checks the registry against the
tree (each statement is a line of the case file, in the order of the list) and, with
``--check-base``, against the registry of the base revision: no piece may be added. A base without
the registry records nothing to compare. This module reads the source tree only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "tests/gates/vertical_slice.toml"
GATE = "vertical-slice"
PIECE_KEYS = {"id", "statement", "refusal", "owner"}


class RegistryError(ValueError):
    """The registry breaks a rule of the slice's gate."""


def load(text: str) -> dict:
    data = tomllib.loads(text)
    if data.get("schema_version") != 1 or data.get("gate") != GATE or "case" not in data:
        raise RegistryError("%s: schema_version 1, gate %r and case are required"
                            % (REGISTRY, GATE))
    pieces = data.get("piece", [])
    for piece in pieces:
        if set(piece) != PIECE_KEYS:
            raise RegistryError("a [[piece]] has id, statement, refusal and owner: %r" % (piece,))
        if ": " not in piece["refusal"]:
            raise RegistryError("the refusal of piece %s is 'Type: message': %r"
                                % (piece["id"], piece["refusal"]))
    ids = [piece["id"] for piece in pieces]
    if len(ids) != len(set(ids)):
        raise RegistryError("a piece id is listed twice")
    return {"case": data["case"], "piece": pieces}


def read(root: Path = ROOT) -> dict:
    return load((root / REGISTRY).read_text(encoding="utf-8"))


def lines_of(registry: dict, root: Path = ROOT) -> list:
    """The line of the case file of each piece, in the order of the list."""
    case = root / registry["case"]
    if not case.is_file():
        raise RegistryError("the case file %s does not exist" % registry["case"])
    text = [line.strip() for line in case.read_text(encoding="utf-8").splitlines()]
    found = []
    for piece in registry["piece"]:
        where = [number for number, line in enumerate(text, 1) if line == piece["statement"]]
        if len(where) != 1:
            raise RegistryError("the statement of piece %s is %d lines of %s, not one: %r"
                                % (piece["id"], len(where), registry["case"], piece["statement"]))
        found.append(where[0])
    return found


def check(root: Path = ROOT) -> list:
    registry = read(root)
    lines = lines_of(registry, root)
    if lines != sorted(lines):
        return ["the pieces are not in the order of the case file: lines %s" % (lines,)]
    return []


def _shown(root: Path, revision: str) -> str | None:
    shown = subprocess.run(["git", "-C", str(root), "show", "%s:%s" % (revision, REGISTRY)],
                           capture_output=True, text=True)
    if shown.returncode != 0:
        exists = subprocess.run(["git", "-C", str(root), "cat-file", "-e",
                                 revision + "^{commit}"], capture_output=True)
        if exists.returncode != 0:
            raise RegistryError("the base revision %s is not in the repository" % revision)
        return None
    return shown.stdout


def grown(current: dict, base: dict) -> list:
    """The pieces that the registry records beyond its base (MG6)."""
    known = {piece["id"] for piece in base["piece"]}
    return ["[[piece]] %s is not in the base" % piece["id"] for piece in current["piece"]
            if piece["id"] not in known]


def check_base(revision: str, root: Path = ROOT) -> list:
    text = _shown(root, revision)
    if text is None:
        return []
    return grown(read(root), load(text))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("check", help="check the registry against the tree")
    command.add_argument("--check-base", metavar="REVISION",
                         help="also fail when the registry records more than at REVISION (MG6)")
    command.add_argument("--source", default=str(ROOT), help="the root of the tree")
    arguments = parser.parse_args(argv)
    root = Path(arguments.source).resolve()
    try:
        problems = check(root)
        if arguments.check_base:
            problems += check_base(arguments.check_base, root)
    except RegistryError as error:
        problems = [str(error)]
    for problem in problems:
        print("vertical slice: %s" % problem, file=sys.stderr)
    if not problems:
        print("vertical slice: %s lists %d piece(s) in the order of the case file"
              % (REGISTRY, len(read(root)["piece"])))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
