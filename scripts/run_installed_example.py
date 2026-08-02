#!/usr/bin/env python3
"""Run one release example only after authenticating its installed PoPS runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MARKER = "PoPS release runtime | native_sha256="


class InstalledExampleError(RuntimeError):
    """The example is not bound to the expected installed native runtime."""


def _outside_checkout(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    raise InstalledExampleError("%s must be outside the checkout: %s" % (label, resolved))


def verify_installed_runtime(expected_sha256: str) -> str:
    """Import PoPS once and return the authenticated native extension digest."""
    if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256):
        raise InstalledExampleError("expected native sha256 is malformed")

    import pops
    from pops import _pops

    _outside_checkout(Path(pops.__file__), label="installed PoPS package")
    extension = _outside_checkout(
        Path(_pops.__file__), label="installed PoPS native extension")
    digest = hashlib.sha256(extension.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise InstalledExampleError(
            "installed native extension does not match signed release runtime")
    if pops.__version__ != _pops.__version__:
        raise InstalledExampleError("installed Python and native versions disagree")
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--example", required=True, type=Path)
    parser.add_argument("example_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        digest = verify_installed_runtime(args.runtime_sha256)
        example = args.example.resolve()
        if not example.is_file():
            raise InstalledExampleError("release example is not a readable file: %s" % example)
        try:
            example.relative_to(ROOT)
        except ValueError as exc:
            raise InstalledExampleError(
                "release example must belong to this checkout: %s" % example) from exc
        forwarded = list(args.example_args)
        if forwarded[:1] == ["--"]:
            forwarded.pop(0)
        print(RUNTIME_MARKER + digest, flush=True)
        sys.argv = [str(example), *forwarded]
        runpy.run_path(str(example), run_name="__main__")
        return 0
    except (InstalledExampleError, OSError, ValueError) as exc:
        print("installed example failed: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
