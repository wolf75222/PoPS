#!/usr/bin/env python3
"""Single, fail-closed authority for PoPS CI route decisions and gate verdicts.

GitHub job ``if`` expressions and the required-check aggregator must consume the same route
decision.  Keeping that decision in this stdlib-only module makes PR path, label, push and release
cases executable in the source-only architecture lane instead of duplicating shell conditionals.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class RouteDecision:
    """Required jobs for one workflow event."""

    full: bool
    cpp_required: bool
    python_required: bool
    architecture_required: bool
    mpi_required: bool
    openmp_required: bool

    def outputs(self) -> dict[str, str]:
        return {
            name: "true" if value else "false"
            for name, value in self.__dict__.items()
        }


def decide_routes(
    *,
    event_name: str,
    cpp_paths: bool | str = False,
    python_paths: bool | str = False,
    architecture_paths: bool | str = False,
    mpi_paths: bool | str = False,
    full_paths: bool | str = False,
    ci_kokkos: bool | str = False,
    ci_full: bool | str = False,
    force_full: bool | str = False,
) -> RouteDecision:
    """Return the only route decision used by jobs and their final aggregator."""
    if not event_name:
        raise RouteModeError("event_name must be non-empty")
    cpp = parse_bool(cpp_paths, name="cpp_paths")
    python = parse_bool(python_paths, name="python_paths")
    architecture = parse_bool(architecture_paths, name="architecture_paths")
    mpi = parse_bool(mpi_paths, name="mpi_paths")
    full_path = parse_bool(full_paths, name="full_paths")
    full_label = parse_bool(ci_full, name="ci_full")
    forced_full = parse_bool(force_full, name="force_full")
    force_kokkos = parse_bool(ci_kokkos, name="ci_kokkos") or full_label or forced_full

    is_pr = event_name == "pull_request"
    full = (
        event_name in {"schedule", "workflow_dispatch", "workflow_call"}
        or forced_full
        or (event_name == "push" and full_path)
        or (is_pr and full_label)
    )
    cpp_required = not is_pr or cpp or force_kokkos
    python_required = not is_pr or cpp or python or force_kokkos
    # Architecture tests inspect Python, bindings, native runtime sources and installed headers.
    architecture_required = (
        not is_pr or cpp or python or architecture or force_kokkos
    )
    return RouteDecision(
        full=full,
        cpp_required=cpp_required,
        python_required=python_required,
        architecture_required=architecture_required,
        mpi_required=full or mpi,
        openmp_required=full,
    )


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


def _add_bool_argument(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument("--" + name.replace("_", "-"), default="false")


def _write_outputs(path: Path, decision: RouteDecision) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in decision.outputs().items():
            output.write("%s=%s\n" % (name, value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide = subparsers.add_parser("decide")
    decide.add_argument("--event-name", required=True)
    for name in (
        "cpp_paths", "python_paths", "architecture_paths", "mpi_paths", "full_paths",
        "ci_kokkos", "ci_full", "force_full",
    ):
        _add_bool_argument(decide, name)
    decide.add_argument("--github-output", type=Path, required=True)

    check = subparsers.add_parser("check")
    check.add_argument(
        "--gate", action="append", nargs=3, metavar=("NAME", "RESULT", "REQUIRED"),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "decide":
            decision = decide_routes(
                event_name=args.event_name,
                cpp_paths=args.cpp_paths,
                python_paths=args.python_paths,
                architecture_paths=args.architecture_paths,
                mpi_paths=args.mpi_paths,
                full_paths=args.full_paths,
                ci_kokkos=args.ci_kokkos,
                ci_full=args.ci_full,
                force_full=args.force_full,
            )
            _write_outputs(args.github_output, decision)
            print("CI routes: " + ", ".join(
                "%s=%s" % item for item in decision.outputs().items()
            ))
        else:
            validate_gate_results(args.gate)
            print("CI gate results are coherent with the required routes")
    except RouteModeError as error:
        parser = _parser()
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
