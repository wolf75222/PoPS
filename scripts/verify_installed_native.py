#!/usr/bin/env python3
"""Select and verify one exact installed PoPS native dimension and its capabilities."""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from write_native_variant_manifest import (  # noqa: E402
    NativeVariantManifestError,
    exact_dimensions,
    load_manifest,
    sha256_file,
)


class InstalledNativeVerificationError(RuntimeError):
    """The installed extension does not implement the requested native contract."""


def verify_installed_native(
    *,
    expect_dimension: int,
    expect_mpi: bool | None = None,
    expect_parallel_hdf5: bool | None = None,
    selector: Callable[[int], ModuleType] | None = None,
) -> Path:
    """Select only ``Dim=N`` and authenticate its manifest row and compiled facts."""
    try:
        dimension = exact_dimensions(
            (expect_dimension,), where="installed native expectation"
        )[0]
    except NativeVariantManifestError as exc:
        raise InstalledNativeVerificationError(str(exc)) from exc
    if expect_parallel_hdf5 is True and expect_mpi is not True:
        raise InstalledNativeVerificationError(
            "parallel HDF5 verification requires the native MPI contract")
    if selector is None:
        from pops._native_selector import select_native_dimension

        selector = select_native_dimension
    native = selector(dimension)
    if getattr(native, "__name__", None) != "pops._pops":
        raise InstalledNativeVerificationError(
            "selected native leaf was not loaded under the logical pops._pops name"
        )
    actual_dimension = getattr(native, "__native_dimension__", None)
    if type(actual_dimension) is not int or actual_dimension != dimension:
        raise InstalledNativeVerificationError(
            "selected native leaf authenticates Dim=%r, expected Dim=%d"
            % (actual_dimension, dimension)
        )
    origin_value = getattr(native, "__file__", None)
    if not isinstance(origin_value, str) or not origin_value:
        raise InstalledNativeVerificationError(
            "pops._pops has no concrete installed extension origin")
    origin = Path(origin_value).resolve()
    if not origin.is_file():
        raise InstalledNativeVerificationError(
            "pops._pops extension does not exist: %s" % origin)
    if not any(str(origin).endswith(suffix)
               for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise InstalledNativeVerificationError(
            "pops._pops origin is not a native extension: %s" % origin)
    if origin.parent.name != "dim%d" % dimension or origin.parent.parent.name != "_native" \
            or origin.name not in {
                "_pops" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES
            }:
        raise InstalledNativeVerificationError(
            "pops._pops did not originate from the exact _native/dim%d leaf: %s"
            % (dimension, origin)
        )

    manifest = origin.parent.parent / "variants.json"
    try:
        rows = load_manifest(manifest, verify_files=True, verify_hashes=True)
    except NativeVariantManifestError as exc:
        raise InstalledNativeVerificationError(str(exc)) from exc
    row = next((item for item in rows if item["dimension"] == dimension), None)
    if row is None:
        raise InstalledNativeVerificationError(
            "variants.json does not declare the selected Dim=%d leaf" % dimension
        )
    manifest_origin = (manifest.parent / row["path"]).resolve()
    if manifest_origin != origin:
        raise InstalledNativeVerificationError(
            "selected Dim=%d path disagrees with variants.json" % dimension
        )
    actual_version = getattr(native, "__version__", None)
    if actual_version != row["version"]:
        raise InstalledNativeVerificationError(
            "selected native version disagrees with variants.json"
        )
    abi_provider = getattr(native, "abi_key", None)
    actual_abi = abi_provider() if callable(abi_provider) else None
    if actual_abi != row["abi_key"]:
        raise InstalledNativeVerificationError(
            "selected native ABI key disagrees with variants.json"
        )

    has_mpi = getattr(native, "__has_mpi__", None)
    if type(has_mpi) is not bool or has_mpi is not row["has_mpi"]:
        raise InstalledNativeVerificationError(
            "selected native MPI fact disagrees with variants.json"
        )
    has_kokkos = getattr(native, "__has_kokkos__", None)
    if type(has_kokkos) is not bool or has_kokkos is not row["has_kokkos"]:
        raise InstalledNativeVerificationError(
            "selected native Kokkos fact disagrees with variants.json"
        )
    if expect_mpi is not None and has_mpi is not expect_mpi:
        requested = "MPI" if expect_mpi else "serial"
        raise InstalledNativeVerificationError(
            "the installed extension does not expose the requested native %s backend"
            % requested)

    has_parallel_hdf5 = getattr(native, "__has_parallel_hdf5__", None)
    if expect_parallel_hdf5 is not None and has_parallel_hdf5 is not expect_parallel_hdf5:
        requested = "parallel" if expect_parallel_hdf5 else "non-parallel"
        raise InstalledNativeVerificationError(
            "the installed extension does not expose the requested %s HDF5 backend"
            % requested)

    if expect_parallel_hdf5 is True:
        capability_provider = getattr(native, "_parallel_hdf5_capability", None)
        if not callable(capability_provider):
            raise InstalledNativeVerificationError(
                "the installed extension lacks its parallel HDF5 capability provider")
        capability = capability_provider()
        required = {
            "available", "hdf5_version", "reason", "communicator", "implementation",
        }
        if type(capability) is not dict or set(capability) != required:
            raise InstalledNativeVerificationError(
                "the installed parallel HDF5 capability report is malformed")
        if capability["available"] is not True:
            raise InstalledNativeVerificationError(
                "the installed parallel HDF5 capability is not available: %s"
                % capability["reason"])
        if not isinstance(capability["hdf5_version"], str) \
                or not capability["hdf5_version"]:
            raise InstalledNativeVerificationError(
                "the installed parallel HDF5 runtime has no version identity")
        if capability["communicator"] != "explicit native MPI communicator" \
                or capability["implementation"] != "C++ HDF5 C API":
            raise InstalledNativeVerificationError(
                "the installed parallel HDF5 provider is not the explicit native communicator contract")

    return origin


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-dim", required=True, type=int, choices=(1, 2, 3),
        help="the one native specialization this process is allowed to select",
    )
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--expect-mpi", action="store_true")
    backend.add_argument("--expect-serial", action="store_true")
    parser.add_argument("--expect-parallel-hdf5", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.expect_parallel_hdf5 and not args.expect_mpi:
        parser.error("--expect-parallel-hdf5 requires --expect-mpi")
    expected_mpi = True if args.expect_mpi else (False if args.expect_serial else None)
    expected_hdf5 = True if args.expect_parallel_hdf5 else (
        False if args.expect_serial else None)
    try:
        origin = verify_installed_native(
            expect_dimension=args.expect_dim,
            expect_mpi=expected_mpi,
            expect_parallel_hdf5=expected_hdf5,
        )
    except Exception as error:
        print(
            "ERROR: installed PoPS native verification failed: %s" % error,
            file=sys.stderr,
        )
        return 1
    if args.json:
        native = sys.modules["pops._pops"]
        print(json.dumps({
            "dimension": args.expect_dim,
            "has_kokkos": native.__has_kokkos__,
            "has_mpi": native.__has_mpi__,
            "has_parallel_hdf5": native.__has_parallel_hdf5__,
            "native_extension": str(origin),
            "sha256": sha256_file(origin),
        }, sort_keys=True))
    else:
        print("installed native Dim=%d extension: %s" % (args.expect_dim, origin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
