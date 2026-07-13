#!/usr/bin/env python3
"""Ad-hoc sign the exact installed ``pops._pops`` extension before importing it.

The locator deliberately does not import :mod:`pops`: importing the package is precisely the
operation macOS may kill when a wheel rewrite has invalidated the extension signature.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


class CodesignError(RuntimeError):
    """The installed native extension could not be located, signed, or authenticated."""


def _find_child_spec(fullname: str, locations: Sequence[str]):
    """Run the normal meta-path resolution with a known parent path, without importing parent."""
    for finder in sys.meta_path:
        find_spec = getattr(finder, "find_spec", None)
        if not callable(find_spec):
            continue
        try:
            spec = find_spec(fullname, locations)
        except ModuleNotFoundError:
            continue
        if spec is not None:
            return spec
    return None


def locate_imported_pops_extensions() -> tuple[Path, ...]:
    """Resolve the extension origins used by a clean ``import pops`` without importing it."""
    package = importlib.util.find_spec("pops")
    if package is None:
        return ()
    locations = package.submodule_search_locations
    if not locations:
        raise CodesignError("installed 'pops' is not a package; cannot resolve pops._pops")
    extension = _find_child_spec("pops._pops", locations)
    if extension is None or not extension.origin:
        raise CodesignError(
            "installed 'pops' package has no importable pops._pops native extension")
    origin = Path(extension.origin).resolve()
    if not origin.is_file():
        raise CodesignError("resolved pops._pops extension does not exist: %s" % origin)
    if not any(str(origin).endswith(suffix)
               for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise CodesignError(
            "resolved pops._pops origin is not a native extension: %s" % origin)
    return (origin,)


def _checked_codesign(command: Sequence[str], *, action: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise CodesignError("%s failed (exit %d): %s" % (
            action, result.returncode, detail))
    return result


def codesign_imported_extensions(*, if_present: bool = False) -> tuple[Path, ...]:
    """Sign and verify every extension a clean ``import pops`` will load on Darwin."""
    if sys.platform != "darwin":
        return ()
    extensions = locate_imported_pops_extensions()
    if not extensions:
        if if_present:
            return ()
        raise CodesignError(
            "installed 'pops' package was not found after build/install; refusing to import")
    codesign = shutil.which("codesign")
    if not codesign:
        raise CodesignError("Darwin requires 'codesign', but it is not available on PATH")
    for extension in extensions:
        _checked_codesign(
            (codesign, "--force", "--sign", "-", str(extension)),
            action="ad-hoc signing %s" % extension)
        _checked_codesign(
            (codesign, "--verify", "--strict", "--verbose=2", str(extension)),
            action="signature verification %s" % extension)
        inspection = _checked_codesign(
            (codesign, "--display", "--verbose=4", str(extension)),
            action="signature inspection %s" % extension)
        evidence = "%s\n%s" % (inspection.stdout, inspection.stderr)
        if "Signature=adhoc" not in evidence:
            raise CodesignError(
                "codesign verification succeeded but the signature is not ad hoc: %s"
                % extension)
    return extensions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--if-present", action="store_true",
        help="skip only when the pops package is absent (a present package without _pops fails)")
    args = parser.parse_args(argv)
    try:
        extensions = codesign_imported_extensions(if_present=args.if_present)
    except CodesignError as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    if sys.platform == "darwin":
        if extensions:
            for extension in extensions:
                print("codesign: verified ad-hoc signature: %s" % extension)
        else:
            print("codesign: pops is not installed yet; nothing to sign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
