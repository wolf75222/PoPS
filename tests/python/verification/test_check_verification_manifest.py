"""Fail-closed checker for verification/manifest.toml (plan §5)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_verification_manifest.py"
MANIFEST = REPO_ROOT / "verification" / "manifest.toml"

# Plan §5 example: same header as the current manifest plus one CP-02 case.
PLAN_SECTION_5_EXAMPLE = """\
schema = "pops.verification.manifest.v1"
repository = "wolf75222/PoPS"
max_nodes = 2

[current_capabilities]
exact_native_dimension = true
cartesian_system_runtime = true
polar_system_runtime = false
amr_total_levels_baseline = 3
amr_refinement_ratios_baseline = [2, 2]
hdf5_requires_mpi = true

[[case]]
id = "CP-02"
path = "verification/cases/euler_poisson/langmuir_cold/run.py"
name = "Cold Langmuir wave"
verification_kind = "code-verification"
evidence_status = "required"
physics = ["continuity", "momentum", "poisson", "electrostatic_source"]
oracle = "linear_eigenmode_and_closed_form"
native_dimensions = [1, 2]
execution_spaces = ["KokkosSerial", "KokkosOpenMP", "KokkosCuda"]
mpi_modes = ["off", "on"]
suites = ["pr", "nightly", "weekly", "release", "two_node"]
requires = [
  "public_case_pipeline",
  "cartesian_layout",
  "poisson",
  "field_at_program_stage",
]

[case.resources.pr]
nodes = 1
mpi_ranks = 1
omp_threads = 1
resolutions = [32, 64, 128]

[case.resources.two_node]
nodes = [1, 2]
mpi_ranks_per_node = [1, 2, 4]
gpus_per_node = [1, 2, 4]
max_wall_seconds = 3600

[case.acceptance]
spatial_order_min = 1.8
temporal_order_min = 1.8
poisson_relative_residual_max = 1.0e-10
finite = true
charge_conservation = true
"""


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_current_repo_manifest_exits_zero():
    result = _run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert result.stdout.count("\n") == 1


def test_plan_section_5_example_exits_zero(tmp_path: Path):
    path = tmp_path / "manifest.toml"
    path.write_text(PLAN_SECTION_5_EXAMPLE, encoding="utf-8")
    result = _run("--manifest", str(path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_max_nodes_three_exits_one_and_reports_failure(tmp_path: Path):
    text = MANIFEST.read_text(encoding="utf-8").replace("max_nodes = 2", "max_nodes = 3", 1)
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    result = _run("--manifest", str(path))
    assert result.returncode == 1
    assert "max_nodes" in result.stderr


def test_missing_file_exits_one(tmp_path: Path):
    missing = tmp_path / "does-not-exist.toml"
    result = _run("--manifest", str(missing))
    assert result.returncode == 1
    assert result.stderr.strip()
