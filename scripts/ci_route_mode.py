#!/usr/bin/env python3
"""Fail-closed gate verdicts for the shared PoPS CI execution plan.

scripts/ci_plan.py owns job selection. The workflow passes that plan's required
flags and actual job results to this stdlib-only check before reporting success.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence


class RouteModeError(RuntimeError):
    """A route input or job result cannot produce a trustworthy CI verdict."""


def parse_bool(value: bool | str, *, name: str) -> bool:
    """Parse the exact lowercase booleans emitted by GitHub expressions."""
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise RouteModeError("%s must be exactly true or false, got %r" % (name, value))


def validate_gate_result(name: str, result: str, required: bool | str) -> None:
    """Accept an optional skip, but require success for every routed job."""
    is_required = parse_bool(required, name="%s.required" % name)
    if is_required:
        if result != "success":
            raise RouteModeError(
                "%s was required but its result is %r" % (name, result)
            )
        return
    if result not in {"success", "skipped"}:
        raise RouteModeError(
            "%s was optional but failed, was cancelled or timed out: %r" % (name, result)
        )


def validate_gate_results(rows: Iterable[Sequence[str]]) -> None:
    """Validate ``(name, result, required)`` rows in declaration order."""
    for row in rows:
        if len(row) != 3:
            raise RouteModeError("one --gate requires NAME RESULT REQUIRED")
        validate_gate_result(row[0], row[1], row[2])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check")
    check.add_argument(
        "--gate", action="append", nargs=3, metavar=("NAME", "RESULT", "REQUIRED"),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_gate_results(args.gate)
        print("CI gate results are coherent with the required routes")
    except RouteModeError as error:
        parser = _parser()
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
