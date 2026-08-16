#!/usr/bin/env python3
"""Ad-hoc sign an explicit installed set of PoPS native variant leaves.

The locator deliberately does not import :mod:`pops`: importing the package is precisely the
operation macOS may kill when a wheel rewrite has invalidated an extension signature.  Variant
paths come only from ``pops/_native/variants.json``; no root extension or filesystem fallback is
eligible.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    exact_dimensions,
    load_manifest,
    sha256_file,
    write_manifest_atomic,
)


CODESIGN_EVIDENCE_SCHEMA_VERSION = 2


class CodesignError(RuntimeError):
    """The installed native extension could not be located, signed, or authenticated."""


@dataclass(frozen=True)
class InstalledNativeVariant:
    dimension: int
    path: Path
    row: dict[str, Any]


def _installed_manifest(*, if_present: bool) -> Path | None:
    """Resolve the one installed manifest without importing the package."""
    package = importlib.util.find_spec("pops")
    if package is None:
        if if_present:
            return None
        raise CodesignError(
            "installed 'pops' package was not found after build/install; refusing to import"
        )
    locations = package.submodule_search_locations
    if not locations:
        raise CodesignError("installed 'pops' is not a package; cannot resolve native variants")
    manifests = {
        (Path(location) / "_native" / "variants.json").resolve()
        for location in locations
        if (Path(location) / "_native" / "variants.json").is_file()
    }
    if len(manifests) != 1:
        raise CodesignError(
            "installed 'pops' package must expose exactly one _native/variants.json; found %d"
            % len(manifests)
        )
    return next(iter(manifests))


def locate_installed_pops_variants(
    expected_dimensions: Sequence[int], *, if_present: bool = False
) -> tuple[InstalledNativeVariant, ...]:
    """Resolve only explicitly requested manifest leaves, without importing :mod:`pops`."""
    dimensions = exact_dimensions(expected_dimensions, where="codesign expected dimensions")
    manifest = _installed_manifest(if_present=if_present)
    if manifest is None:
        return ()
    rows = load_manifest(manifest, verify_files=True, verify_hashes=False)
    available = {row["dimension"] for row in rows}
    missing = set(dimensions) - available
    if missing:
        raise CodesignError(
            "installed native manifest lacks explicitly requested dimensions: %s"
            % tuple(sorted(missing))
        )
    all_variants = tuple(
        InstalledNativeVariant(
            dimension=row["dimension"],
            path=(manifest.parent / PurePosixPath(row["path"])).resolve(),
            row=row,
        )
        for row in rows
    )
    declared_paths = {variant.path for variant in all_variants}
    unmanifested = []
    for candidate in manifest.parent.glob("**/_pops*"):
        if candidate.is_file() and candidate.resolve() not in declared_paths:
            unmanifested.append(candidate.resolve())
    if unmanifested:
        raise CodesignError(
            "installed package contains an unmanifested native extension: %s"
            % ", ".join(str(path) for path in sorted(unmanifested))
        )
    return tuple(
        variant for variant in all_variants if variant.dimension in dimensions
    )


def _checked_codesign(command: Sequence[str], *, action: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise CodesignError("%s failed (exit %d): %s" % (
            action, result.returncode, detail))
    return result


def _has_valid_adhoc_signature(codesign: str, extension: Path) -> bool:
    """Return whether ``extension`` already carries the release signature policy.

    Release validation must not rewrite bytes which came from the retained wheel: those are the
    bytes eventually published.  Probe first and only repair an absent/invalid signature.  The
    release preflight separately refuses a repair which changes the retained native-member digest.
    """

    verification = subprocess.run(
        (codesign, "--verify", "--strict", "--verbose=2", str(extension)),
        text=True,
        capture_output=True,
        check=False,
    )
    if verification.returncode != 0:
        return False
    inspection = subprocess.run(
        (codesign, "--display", "--verbose=4", str(extension)),
        text=True,
        capture_output=True,
        check=False,
    )
    if inspection.returncode != 0:
        return False
    return "Signature=adhoc" in "%s\n%s" % (inspection.stdout, inspection.stderr)


def codesign_imported_extensions(
    expected_dimensions: Sequence[int], *, if_present: bool = False
) -> tuple[InstalledNativeVariant, ...]:
    """Sign every explicitly expected manifest leaf and publish its final digest."""
    dimensions = exact_dimensions(expected_dimensions, where="codesign expected dimensions")
    if sys.platform != "darwin":
        return ()
    variants = locate_installed_pops_variants(dimensions, if_present=if_present)
    if not variants:
        return ()
    codesign = shutil.which("codesign")
    if not codesign:
        raise CodesignError("Darwin requires 'codesign', but it is not available on PATH")
    for variant in variants:
        extension = variant.path
        if _has_valid_adhoc_signature(codesign, extension):
            # Wheel assembly may rewrite Mach-O load commands and produce a new valid ad-hoc
            # signature after the build-tree manifest was generated.  Preserve those verified
            # bytes and authenticate their final digest below.  Release builds separately prove
            # the retained wheel and installed copy byte-for-byte before reaching this helper, so
            # this refresh cannot hide a mutation of a retained release artifact.
            continue
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
    manifest = variants[0].path.parents[1] / "variants.json"
    current_rows = load_manifest(manifest, verify_files=True, verify_hashes=False)
    selected = {variant.dimension: variant for variant in variants}
    final_rows = []
    for current in current_rows:
        row = dict(current)
        variant = selected.get(row["dimension"])
        if variant is not None:
            row["sha256"] = sha256_file(variant.path)
        final_rows.append(row)
    write_manifest_atomic(manifest, final_rows)
    authenticated = load_manifest(
        manifest,
        verify_files=True,
        verify_hashes=True,
    )
    by_dimension = {variant.dimension: variant.path for variant in variants}
    return tuple(
        InstalledNativeVariant(row["dimension"], by_dimension[row["dimension"]], row)
        for row in authenticated if row["dimension"] in by_dimension
    )


def codesign_evidence(variants: Sequence[InstalledNativeVariant]) -> dict[str, Any]:
    """Describe the exact post-sign extension bytes authenticated by this process."""
    return {
        "schema_version": CODESIGN_EVIDENCE_SCHEMA_VERSION,
        "platform": sys.platform,
        "extensions": [
            {
                "dimension": variant.dimension,
                "path": str(variant.path.resolve()),
                "sha256": sha256_file(variant.path),
                "signature": "adhoc",
            }
            for variant in variants
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--if-present", action="store_true",
        help="skip only when pops is absent (a present package without the manifest/set fails)")
    parser.add_argument(
        "--expect-dim", action="append", required=True, type=int, choices=(1, 2, 3),
        help="variant leaves to sign; repeat to sign several explicit dimensions",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print machine-authenticated post-sign paths and hashes")
    args = parser.parse_args(argv)
    try:
        variants = codesign_imported_extensions(
            args.expect_dim, if_present=args.if_present
        )
        evidence = codesign_evidence(variants)
    except (CodesignError, NativeVariantManifestError, OSError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(evidence, sort_keys=True))
        return 0
    if sys.platform == "darwin":
        if variants:
            for variant in variants:
                print(
                    "codesign: verified Dim=%d ad-hoc signature: %s"
                    % (variant.dimension, variant.path)
                )
        else:
            print("codesign: pops is not installed yet; nothing to sign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
