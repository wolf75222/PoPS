"""Stable CLI entry point for verification planning (plan §31)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "run_verification.py"
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


def test_max_nodes_three_exits_one_and_does_not_write_plan(tmp_path: Path):
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "3",
        "--output",
        str(output),
    )
    assert result.returncode == 1
    assert "two-node" in result.stderr.lower() or "two node" in result.stderr.lower()
    assert not (output / "plan.json").exists()
    assert not output.exists()


def test_valid_pr_plan_has_empty_cases(tmp_path: Path):
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "planned 0 cases"
    plan_path = output / "plan.json"
    assert plan_path.is_file()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["suite"] == "pr"
    assert plan["dimensions"] == [1]
    assert plan["max_nodes"] == 2
    assert Path(plan["manifest"]) == MANIFEST.resolve()
    assert plan["cases"] == []


def test_invalid_suite_exits_one(tmp_path: Path):
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "not-a-suite",
        "--dimensions",
        "1",
        "--max-nodes",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 1
    assert not (output / "plan.json").exists()


@pytest.mark.parametrize("dimensions", ["4", ""])
def test_invalid_dimensions_exits_one(tmp_path: Path, dimensions: str):
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        dimensions,
        "--max-nodes",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 1
    assert not (output / "plan.json").exists()


def test_max_nodes_less_than_one_exits_one(tmp_path: Path):
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "0",
        "--output",
        str(output),
    )
    assert result.returncode == 1
    assert not (output / "plan.json").exists()


def test_selects_cases_matching_suite_and_dimensions(tmp_path: Path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(PLAN_SECTION_5_EXAMPLE, encoding="utf-8")
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "2",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "planned 1 cases"
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in plan["cases"]] == ["CP-02"]

    miss = tmp_path / "miss"
    missed = _run(
        "--suite",
        "pr",
        "--dimensions",
        "3",
        "--max-nodes",
        "2",
        "--output",
        str(miss),
        "--manifest",
        str(manifest),
    )
    assert missed.returncode == 0, missed.stderr
    assert missed.stdout.strip() == "planned 0 cases"
    missed_plan = json.loads((miss / "plan.json").read_text(encoding="utf-8"))
    assert missed_plan["cases"] == []
