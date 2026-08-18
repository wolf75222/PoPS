"""Stable CLI entry point for verification planning (plan §31)."""
from __future__ import annotations

import json
import os
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


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.pop("POPS_NATIVE_DIM", None)
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
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


def test_valid_pr_plan_has_dummy_case(tmp_path: Path):
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
    plan_path = output / "plan.json"
    assert plan_path.is_file()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["suite"] == "pr"
    assert plan["dimensions"] == [1]
    assert plan["max_nodes"] == 2
    assert Path(plan["manifest"]) == MANIFEST.resolve()
    ids = [case["id"] for case in plan["cases"]]
    assert "PH-00" in ids
    assert set(ids) >= {
        "PH-00",
        "TR-01",
        "TR-02",
        "PO-01",
        "PO-02",
        "PO-03",
        "PO-07",
        "EU-01",
        "EU-03",
        "TM-01",
        "CP-01",
        "CP-02",
        "CP-03",
        "CP-07",
        "CP-08",
        "CP-12",
        "TM-07",
    }
    assert result.stdout.strip() == f"planned {len(ids)} cases"
    assert {"case_id": "PH-00", "pops_native_dim": 1} in plan["jobs"]


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
    assert plan["jobs"] == [{"case_id": "CP-02", "pops_native_dim": 1}]

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
    assert missed_plan["jobs"] == []


def test_execute_writes_results_and_keeps_plan(tmp_path: Path):
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
        "--execute",
    )
    assert result.returncode in (0, 1), result.stderr
    assert (output / "plan.json").is_file()
    assert (output / "results.json").is_file()
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows
    assert rows[0]["case_id"] == "CP-02"
    assert rows[0]["status"] in {"ok", "skipped", "failed", "no_run_native"}
    assert "planned 1 cases" in result.stdout
    assert "executed" in result.stdout


def test_plan_emits_one_job_per_requested_native_dimension(tmp_path: Path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(PLAN_SECTION_5_EXAMPLE, encoding="utf-8")
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1,2",
        "--max-nodes",
        "2",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["jobs"] == [
        {"case_id": "CP-02", "pops_native_dim": 1},
        {"case_id": "CP-02", "pops_native_dim": 2},
    ]


def test_pops_native_dim_env_matching_request_emits_one_job(tmp_path: Path):
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
        env={"POPS_NATIVE_DIM": "1"},
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["jobs"] == [{"case_id": "CP-02", "pops_native_dim": 1}]


def test_pops_native_dim_cli_overrides_env_for_matching_request(tmp_path: Path):
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
        "--pops-native-dim",
        "1",
        env={"POPS_NATIVE_DIM": "2"},
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["jobs"] == [{"case_id": "CP-02", "pops_native_dim": 1}]


def test_artifact_dim_mismatch_exits_one_without_plan(tmp_path: Path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(PLAN_SECTION_5_EXAMPLE, encoding="utf-8")
    output = tmp_path / "out"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1,2",
        "--max-nodes",
        "2",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        env={"POPS_NATIVE_DIM": "1"},
    )
    assert result.returncode == 1
    combined = f"{result.stderr}\n{result.stdout}"
    assert "POPS_NATIVE_DIM" in combined
    assert "fallback" in combined.lower()
    assert not (output / "plan.json").exists()
