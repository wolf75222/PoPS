"""Use the installed native Kokkos contract for real standalone compiler fixtures."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from pops.codegen import toolchain
from tests.python.support.requirements import require_native_or_skip


def configured_kokkos_toolchain():
    selected = toolchain._native_kokkos_selection()
    compiler = toolchain._native_kokkos_compiler(
        os.environ.get("POPS_TEST_CXX") or os.environ.get("CXX"))
    if selected is None or not compiler:
        require_native_or_skip("real compiler and authenticated Kokkos SDK required",
                               optional_skip=pytest.skip)
        raise AssertionError("native prerequisite guard unexpectedly returned")
    flags, link_flags = toolchain._native_kokkos_flags()
    return compiler, Path(selected[0]), list(flags), list(link_flags)


def mpi_syntax_flags():
    """Read actual MPI wrapper include/options; retain the MPI branch on Serial hosts."""
    requested = os.environ.get("MPI_CXX_COMPILER") or os.environ.get("MPICXX") or "mpicxx"
    wrapper = shutil.which(requested)
    if wrapper is None:
        require_native_or_skip("real MPI SDK compiler wrapper required for MPI-branch syntax",
                               optional_skip=pytest.skip)
        raise AssertionError("native prerequisite guard unexpectedly returned")
    for option in ("--showme:compile", "-compile-info", "-show"):
        result = subprocess.run([wrapper, option], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            flags = shlex.split(result.stdout)
            if option in {"-compile-info", "-show"}:
                # MPICH reports the underlying compiler before its complete wrapper flags.
                flags = flags[1:]
            if flags:
                return flags
    raise RuntimeError("MPI compiler wrapper did not expose its actual SDK compile flags")
