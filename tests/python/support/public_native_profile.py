"""Authenticate exact native ABI, SDK and profile for public integration fixtures.

Historical receipts never authenticate a later execution. Serial carries no MPI
communicator; MPI carries the real authenticated native world from this package.
"""
from __future__ import annotations

import os
from pathlib import Path


def native_profile(*, scope="public native fixture",
                   expected_rank_env="POPS_TEST_PUBLIC_NATIVE_EXPECT_RANKS"):
    import pops
    from pops._generated_release_contract import NATIVE_ABI_VERSION
    from pops._native_selector import select_native_dimension
    from pops.codegen.abi import module_header_signature
    from pops.codegen.toolchain import pops_header_signature, pops_include
    from scripts.verify_installed_native import verify_installed_native

    native = select_native_dimension(2)
    if type(native.__abi_version__) is not int or native.__abi_version__ != NATIVE_ABI_VERSION:
        raise AssertionError(scope + " requires the exact declared native ABI")
    if type(native.__has_mpi__) is not bool:
        raise AssertionError(scope + " requires an exact native MPI capability")
    origin = verify_installed_native(expect_dimension=2, expect_mpi=native.__has_mpi__)
    package = Path(pops.__file__).resolve().parent
    if origin.parents[2] != package:
        raise AssertionError(scope + " native and Python package origins differ")
    if module_header_signature() != pops_header_signature(pops_include()):
        raise AssertionError(scope + " native and SDK header identities differ")
    world = None
    if native.__has_mpi__:
        from pops._native_collectives import require_world

        world = require_world(native.mpi_world())
    expected = os.environ.get(expected_rank_env)
    if expected is not None:
        if expected not in {"1", "2"} or world is None or int(world.size) != int(expected):
            raise AssertionError(scope + " MPI profile differs from its requested world")
    return native, world


def rank_and_size(world):
    return (0, 1) if world is None else (int(world.rank), int(world.size))


__all__ = ["native_profile", "rank_and_size"]
