#!/usr/bin/env python3
"""Verify the exact installed PoPS native extension and its requested capabilities."""
from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType


class InstalledNativeVerificationError(RuntimeError):
    """The installed extension does not implement the requested native contract."""


def verify_installed_native(
    *,
    expect_mpi: bool | None = None,
    expect_parallel_hdf5: bool | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> Path:
    """Import ``pops._pops`` and exercise the capabilities requested by the build."""
    if expect_parallel_hdf5 is True and expect_mpi is not True:
        raise InstalledNativeVerificationError(
            "parallel HDF5 verification requires the native MPI contract")
    native = importer("pops._pops")
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

    has_mpi = getattr(native, "__has_mpi__", None)
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
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--expect-mpi", action="store_true")
    backend.add_argument("--expect-serial", action="store_true")
    parser.add_argument("--expect-parallel-hdf5", action="store_true")
    args = parser.parse_args(argv)
    if args.expect_parallel_hdf5 and not args.expect_mpi:
        parser.error("--expect-parallel-hdf5 requires --expect-mpi")
    expected_mpi = True if args.expect_mpi else (False if args.expect_serial else None)
    expected_hdf5 = True if args.expect_parallel_hdf5 else (
        False if args.expect_serial else None)
    try:
        origin = verify_installed_native(
            expect_mpi=expected_mpi,
            expect_parallel_hdf5=expected_hdf5,
        )
    except Exception as error:
        print(
            "ERROR: installed PoPS native verification failed: %s" % error,
            file=sys.stderr,
        )
        return 1
    print("installed native extension: %s" % origin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
