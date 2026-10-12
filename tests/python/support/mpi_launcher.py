"""Manifest-owned MPI test launcher discovery and fail-fast arguments."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.python.support.requirements import require_mpi_or_skip


def mpi_launcher_or_skip(native_mpi_contract: object, *, purpose: str) -> tuple[str, ...]:
    candidates: list[str] = []
    explicit_launcher = os.environ.get("POPS_MPIEXEC")
    if explicit_launcher:
        if shutil.which(explicit_launcher) is None:
            require_mpi_or_skip(
                f"POPS_MPIEXEC does not resolve to an executable: {explicit_launcher}",
                optional_skip=pytest.skip,
            )
            raise AssertionError("pytest.skip unexpectedly returned")
        candidates.append(explicit_launcher)
    if isinstance(native_mpi_contract, dict):
        compiler = native_mpi_contract.get("compiler")
        if isinstance(compiler, str) and compiler:
            mpi_bin = Path(compiler).parent
            candidates.extend(
                str(mpi_bin / name)
                for name in ("mpiexec", "mpirun", "mpiexec.hydra")
            )
    candidates.extend(("mpiexec", "mpirun", "mpiexec.hydra"))
    launcher = next(
        (
            resolved
            for candidate in candidates
            if (resolved := shutil.which(candidate)) is not None
        ),
        None,
    )
    if launcher is None:
        require_mpi_or_skip(
            purpose + " proof requires mpiexec or mpirun",
            optional_skip=pytest.skip,
        )
        raise AssertionError("pytest.skip unexpectedly returned")
    try:
        version = subprocess.run(
            [launcher, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        require_mpi_or_skip(
            purpose + " launcher is unusable: %s" % exc,
            optional_skip=pytest.skip,
        )
        raise AssertionError("pytest.skip unexpectedly returned") from exc
    if version.returncode != 0:
        require_mpi_or_skip(
            purpose + " launcher --version failed",
            optional_skip=pytest.skip,
        )
        raise AssertionError("pytest.skip unexpectedly returned")
    banner = version.stdout + version.stderr
    fail_fast = (
        ("--mca", "orte_abort_on_non_zero_status", "1")
        if "Open MPI" in banner or "OpenRTE" in banner
        else ()
    )
    return (launcher, *fail_fast)


__all__ = ["mpi_launcher_or_skip"]
