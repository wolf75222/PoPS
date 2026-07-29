"""Serial orchestrator for AMR checkpoint restart across a two-to-one MPI rank change.

This test deliberately does not run inside one long-lived ``mpiexec`` job.  It starts a two-rank
capture process, lets that MPI world terminate, and then starts an independent one-rank restart
process.  That is the only integration shape that proves a persisted checkpoint can cross rank
topologies rather than merely rebuilding ownership inside the original communicator.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.python.support.requirements import require_mpi_or_skip


pytestmark = pytest.mark.mpi
PROBE = Path(__file__).with_name("probe_amr_rank_change_restart.py")
PROCESS_TIMEOUT_SECONDS = 15 * 60


def _launcher_or_skip() -> tuple[str, ...]:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        require_mpi_or_skip(
            "AMR rank-change restart proof requires mpiexec or mpirun",
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
            "AMR rank-change restart launcher is unusable: %s" % exc,
            optional_skip=pytest.skip,
        )
        raise AssertionError("pytest.skip unexpectedly returned") from exc
    if version.returncode != 0:
        require_mpi_or_skip(
            "AMR rank-change restart launcher --version failed",
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


def _run_probe(
    launcher: tuple[str, ...],
    *,
    ranks: int,
    mode: str,
    checkpoint: Path,
    evidence: Path | None = None,
    rematerialized_checkpoint: Path | None = None,
    environment: dict[str, str],
) -> str:
    command = [
        *launcher,
        "-n",
        str(ranks),
        sys.executable,
        "-u",
        str(PROBE),
        mode,
        "--checkpoint",
        str(checkpoint),
    ]
    if evidence is not None:
        command.extend(("--evidence", str(evidence)))
    if rematerialized_checkpoint is not None:
        command.extend(
            ("--rematerialized-checkpoint", str(rematerialized_checkpoint))
        )
    result = subprocess.run(
        command,
        cwd=PROBE.parents[4],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=PROCESS_TIMEOUT_SECONDS,
    )
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if line.strip().startswith("POPS_SKIP:"):
            reason = line.partition("POPS_SKIP:")[2].strip()
            require_mpi_or_skip(reason, optional_skip=pytest.skip)
            raise AssertionError("pytest.skip unexpectedly returned")
    assert result.returncode == 0, (
        "AMR rank-change %s process failed with status %d\n%s"
        % (mode, result.returncode, output)
    )
    return output


def test_amr_checkpoint_restart_rematerializes_two_ranks_onto_one(
    tmp_path: Path,
) -> None:
    try:
        from pops import _pops
    except (ImportError, RuntimeError) as exc:
        require_mpi_or_skip(
            "AMR rank-change restart cannot import its native module: %s" % exc,
            optional_skip=pytest.skip,
        )
        raise AssertionError("pytest.skip unexpectedly returned") from exc
    if getattr(_pops, "__has_mpi__", False) is not True:
        require_mpi_or_skip(
            "AMR rank-change restart requires an MPI-enabled native module",
            optional_skip=pytest.skip,
        )
        raise AssertionError("pytest.skip unexpectedly returned")

    launcher = _launcher_or_skip()
    environment = dict(os.environ)
    # Both source ranks and both later singleton jobs share the exact production artifact cache.
    # The probe's compile-once helper serializes publication within each MPI world.
    cache = tmp_path / "native-cache"
    cache.mkdir()
    environment["POPS_CACHE_DIR"] = str(cache)
    environment["POPS_NATIVE_CACHE_DIR"] = str(cache)

    relaxed = tmp_path / "rank-change-relaxed.npz"
    strict = tmp_path / "rank-change-strict.npz"
    evidence = tmp_path / "rank-change-evidence.npz"
    rematerialized = tmp_path / "rank-change-rematerialized.npz"

    capture_relaxed = _run_probe(
        launcher,
        ranks=2,
        mode="capture-relaxed",
        checkpoint=relaxed,
        evidence=evidence,
        environment=environment,
    )
    assert "PASS capture rematerialized AMR checkpoint on two ranks" in capture_relaxed
    assert relaxed.is_file() and evidence.is_file()

    restart_relaxed = _run_probe(
        launcher,
        ranks=1,
        mode="restart-relaxed",
        checkpoint=relaxed,
        evidence=evidence,
        rematerialized_checkpoint=rematerialized,
        environment=environment,
    )
    assert "PASS restart AMR checkpoint from two ranks onto one rank" in restart_relaxed
    assert rematerialized.is_file()

    capture_strict = _run_probe(
        launcher,
        ranks=2,
        mode="capture-strict",
        checkpoint=strict,
        environment=environment,
    )
    assert "PASS capture strict AMR checkpoint on two ranks" in capture_strict
    assert strict.is_file()

    restart_strict = _run_probe(
        launcher,
        ranks=1,
        mode="restart-strict",
        checkpoint=strict,
        environment=environment,
    )
    assert "PASS bit_identical=True refuses AMR two-to-one restart atomically" in restart_strict
