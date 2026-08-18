"""Static contract for the isolated ROMEO PR-676 foundation jobs."""
from __future__ import annotations

from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[3]
MACHINES = REPO_ROOT / "verification" / "machines"
ENV = MACHINES / "romeo_676_env.sh"
BUILD = MACHINES / "romeo_676_build.sbatch"
SERIAL = MACHINES / "romeo_676_gate_serial.sbatch"
MPI = MACHINES / "romeo_676_gate_mpi.sbatch"
VALIDATE = MACHINES / "romeo_676_validate.py"
SCRIPTS = (ENV, BUILD, SERIAL, MPI, VALIDATE)


def test_romeo_676_scripts_exist_and_are_syntactically_valid() -> None:
    for path in SCRIPTS:
        assert path.is_file(), path
    for path in (ENV, BUILD, SERIAL, MPI):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{path}: {completed.stderr}"


def test_romeo_676_jobs_do_not_enable_nounset_before_sourcing_env() -> None:
    for path in (BUILD, SERIAL, MPI):
        text = path.read_text(encoding="utf-8")
        source_at = text.index("romeo_676_env.sh")
        prefix = text[:source_at]
        assert "set -u" not in prefix
        assert "set -euo" not in prefix


def test_romeo_676_jobs_stay_inside_authorized_romeo_bounds() -> None:
    for path in (BUILD, SERIAL, MPI):
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --account=r250127" in text
        assert "#SBATCH --constraint=x64cpu" in text
        assert "--exclusive" not in text
        assert "gpu" not in text.lower() or "no gpu" in text.lower()
        assert "$HOME/kokkos-x64" not in text
        assert "/scratch_p/rmdraux/pops-verify" not in text
        assert "/scratch_p/rmdraux/pops-676" in text
        assert "kokkos-x64-pic" in text or "romeo_676_env.sh" in text


def test_romeo_676_env_uses_pic_kokkos_and_isolated_scratch_caches() -> None:
    text = ENV.read_text(encoding="utf-8")
    assert "pops-validate/env_x64cpu.sh" in text
    assert "/scratch_p/rmdraux/kokkos-x64-pic" in text
    assert "/scratch_p/rmdraux/pops-676" in text
    assert "POPS_CACHE_DIR" in text
    assert "XDG_CACHE_HOME" in text
    assert "KOKKOS_CPU_ROOT" in text
    assert "$HOME/kokkos-x64" not in text or "must not" in text.lower()


def test_romeo_676_build_keeps_serial_and_mpi_artifacts_distinct() -> None:
    text = BUILD.read_text(encoding="utf-8")
    assert "POPS_NATIVE_DIM" in text
    assert "POPS_USE_MPI" in text
    assert "POPS_USE_HDF5" in text
    assert "build-dim" in text
    assert "verify_installed_native.py" in text
    assert "--nodes=1" in text or "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text


def test_romeo_676_mpi_build_passes_concrete_mpi_include() -> None:
    env = ENV.read_text(encoding="utf-8")
    build = BUILD.read_text(encoding="utf-8")
    assert "MPI_HOME" in env
    assert "include/mpi.h" in env
    assert "MPI_CXX_INCLUDE_DIRS" in build
    assert "MPI_CXX_HEADER_DIR" in build
    assert "MPI_C_INCLUDE_DIRS" in build
    assert "MPI_CXX_LIBRARIES" in build
    assert "MPI_C_LIBRARIES" in build
    assert "libmpi.so" in build


def test_romeo_676_serial_gate_runs_if08_then_eu01() -> None:
    text = SERIAL.read_text(encoding="utf-8")
    assert "run_verification.py" in text
    assert "IF-08" in text
    assert "EU-01" in text
    assert "--mpi-mode off" in text
    assert "romeo_676_validate.py" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text


def test_romeo_676_mpi_gate_launches_if01_with_srun_two_ranks() -> None:
    text = MPI.read_text(encoding="utf-8")
    assert "srun" in text
    assert "IF-01" in text
    assert "--mpi-mode on" in text
    assert "--ntasks=2" in text
    assert "romeo_676_validate.py" in text
    assert "run_verification.py" in text
    assert "#SBATCH --ntasks=2" in text
    nodes = [
        line for line in text.splitlines() if line.startswith("#SBATCH --nodes=")
    ]
    assert nodes
    assert int(nodes[0].split("=", 1)[1]) <= 2
