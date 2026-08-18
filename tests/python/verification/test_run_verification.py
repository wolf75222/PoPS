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
    assert "TR-01" not in ids
    assert set(ids) >= {
        "PH-00",
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
    assert any(
        job["case_id"] == "PH-00" and job["pops_native_dim"] == 1 for job in plan["jobs"]
    )


def test_valid_pr_plan_includes_tr01_only_for_dimension_3(tmp_path: Path):
    output = tmp_path / "out3"
    result = _run(
        "--suite",
        "pr",
        "--dimensions",
        "3",
        "--max-nodes",
        "2",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    ids = [case["id"] for case in plan["cases"]]
    assert "TR-01" in ids
    assert any(
        job["case_id"] == "TR-01" and job["pops_native_dim"] == 3 for job in plan["jobs"]
    )


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
    assert [job["case_id"] for job in plan["jobs"]] == ["CP-02"]
    assert plan["jobs"][0]["pops_native_dim"] == 1

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


def _manifest_header() -> str:
    return """\
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
"""


def _write_case_manifest(
    tmp_path: Path,
    *,
    case_id: str,
    run_py: str,
    evidence_status: str = "required",
    requires: list[str] | None = None,
    mpi_modes: str = '["off"]',
    execution_spaces: str = '["KokkosSerial"]',
    extra_cases: str = "",
) -> tuple[Path, Path]:
    case_dir = tmp_path / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "run.py").write_text(run_py, encoding="utf-8")
    req = requires if requires is not None else []
    req_lit = "[" + ", ".join(f'"{item}"' for item in req) + "]"
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_header()
        + f"""
[[case]]
id = "{case_id}"
path = "{case_dir / "run.py"}"
name = "fixture {case_id}"
verification_kind = "infrastructure"
evidence_status = "{evidence_status}"
physics = []
oracle = "fixture"
native_dimensions = [1]
execution_spaces = {execution_spaces}
mpi_modes = {mpi_modes}
suites = ["pr"]
requires = {req_lit}

[case.resources.pr]
nodes = 1
mpi_ranks = 1
omp_threads = 1
resolutions = [32, 64]

[case.resources.two_node]
nodes = [1, 2]

[case.acceptance]
finite = true
"""
        + extra_cases,
        encoding="utf-8",
    )
    return manifest, case_dir


def _write_authenticated_leaf(tmp_path: Path, *, dimension: int = 1, has_mpi: bool = False) -> Path:
    import hashlib
    import importlib.machinery

    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    root = tmp_path / "native"
    leaf = root / f"dim{dimension}" / f"_pops{suffix}"
    leaf.parent.mkdir(parents=True, exist_ok=True)
    payload = b"fake-exact-rank-leaf"
    leaf.write_bytes(payload)
    row = {
        "dimension": dimension,
        "path": f"dim{dimension}/_pops{suffix}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "version": "1.0.0",
        "abi_key": "abi-test",
        "has_mpi": has_mpi,
        "has_kokkos": True,
    }
    (root / "variants.json").write_text(
        json.dumps({"schema_version": 1, "variants": [row]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def test_execute_without_authenticated_artifact_is_refused(tmp_path: Path):
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="def run_native(request=None, n_cells=None):\n    return None\n",
    )
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
    assert result.returncode == 1
    combined = f"{result.stderr}\n{result.stdout}".lower()
    assert "authenticat" in combined or "exact-rank" in combined or "variant" in combined
    assert not (output / "results.json").exists() or (
        json.loads((output / "results.json").read_text(encoding="utf-8"))
        and all(row.get("status") != "pass" for row in json.loads((output / "results.json").read_text(encoding="utf-8")))
    )


def test_plan_without_native_still_writes_plan(tmp_path: Path):
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="def run_native(request=None, n_cells=None):\n    return None\n",
    )
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
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["jobs"][0]["case_id"] == "IF-08"
    assert plan["jobs"][0]["pops_native_dim"] == 1
    assert not (output / "results.json").exists()


def test_execute_passes_manifest_parameters_to_run_native(tmp_path: Path):
    manifest, case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="""
from pathlib import Path
import json

def run_native(request=None, n_cells=None):
    Path(__file__).with_name("called.json").write_text(
        json.dumps({
            "has_request": request is not None,
            "n_cells": n_cells,
            "mpi_mode": getattr(request, "mpi_mode", None),
            "suite": getattr(request, "suite", None),
            "execution_space": getattr(request, "execution_space", None),
        }),
        encoding="utf-8",
    )
    return {
        "compiler": "not-compiled",
        "build_type": "not-compiled",
        "precision": "not-compiled",
        "block_size": [n_cells or 32],
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": "fixture",
        "cfl": 0.45,
        "final_time": 0.25,
        "gpus": 0,
    }
""",
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 0, result.stderr
    called = json.loads((case_dir / "called.json").read_text(encoding="utf-8"))
    assert called["has_request"] is True
    assert called["n_cells"] == 32
    assert called["suite"] == "pr"
    assert called["execution_space"] == "KokkosSerial"
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "pass"


def test_execute_required_native_unavailable_is_fail(tmp_path: Path):
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="""
class NativeUnavailable(RuntimeError):
    pass

def run_native(request=None, n_cells=None):
    raise NativeUnavailable("kokkos missing")
""",
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 1
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "fail"
    assert rows[0]["status"] not in {"skipped", "ok", "no_run_native"}


def test_execute_required_missing_runner_is_fail(tmp_path: Path):
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="VALUE = 1\n",
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 1
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "fail"


def test_execute_capability_gated_unavailable_is_not_supported(tmp_path: Path):
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="GE-01",
        evidence_status="capability-gated",
        requires=["polar_system_runtime"],
        run_py="""
class NativeUnavailable(RuntimeError):
    pass

def run_native(request=None, n_cells=None):
    raise NativeUnavailable("polar runtime is absent")
""",
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "not-supported"


def test_execute_writes_schema_valid_job_artifacts_and_report(tmp_path: Path):
    from jsonschema import Draft202012Validator

    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py=(
            "def run_native(request=None, n_cells=None):\n"
            "    return {\n"
            '        "compiler": "not-compiled",\n'
            '        "build_type": "not-compiled",\n'
            '        "precision": "not-compiled",\n'
            '        "block_size": [n_cells or 32],\n'
            '        "amr_total_levels": 1,\n'
            '        "refinement_ratio": 2,\n'
            '        "subcycling": False,\n'
            '        "time_program": "fixture",\n'
            '        "cfl": 0.45,\n'
            '        "final_time": 0.25,\n'
            '        "gpus": 0,\n'
            "    }\n"
        ),
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 0, result.stderr
    job_dir = output / "IF-08" / "dim1-KokkosSerial-off"
    for name in ("resolved_case.json", "metrics.json", "provenance.json"):
        assert (job_dir / name).is_file(), name
    metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    Draft202012Validator(
        json.loads((REPO_ROOT / "schemas" / "verification_metrics.v1.json").read_text())
    ).validate(metrics)
    Draft202012Validator(
        json.loads((REPO_ROOT / "schemas" / "verification_provenance.v1.json").read_text()),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(provenance)
    Draft202012Validator(
        json.loads((REPO_ROOT / "schemas" / "verification_report.v1.json").read_text())
    ).validate(summary)
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] in {"pass", "fail", "not-supported", "not-run"}


def test_cases_and_mpi_mode_and_execution_space_filters(tmp_path: Path):
    extra = """
[[case]]
id = "IF-01"
path = "verification/cases/infrastructure/mpi_invariance/run.py"
name = "MPI decomposition invariance"
verification_kind = "infrastructure"
evidence_status = "required"
physics = ["transport"]
oracle = "fixture"
native_dimensions = [1]
execution_spaces = ["KokkosSerial"]
mpi_modes = ["on"]
suites = ["pr"]
requires = ["mpi"]

[case.resources.pr]
nodes = 1

[case.resources.two_node]
nodes = [1, 2]

[case.acceptance]
finite = true
"""
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="def run_native(request=None, n_cells=None):\n    return None\n",
        extra_cases=extra,
    )
    output = tmp_path / "out"
    filtered = _run(
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
        "--cases",
        "IF-08",
        "--mpi-mode",
        "off",
        "--execution-space",
        "KokkosSerial",
    )
    assert filtered.returncode == 0, filtered.stderr
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert [job["case_id"] for job in plan["jobs"]] == ["IF-08"]
    assert plan["jobs"][0]["mpi_mode"] == "off"
    assert plan["jobs"][0]["execution_space"] == "KokkosSerial"

    miss = tmp_path / "miss"
    skipped = _run(
        "--suite",
        "pr",
        "--dimensions",
        "1",
        "--max-nodes",
        "2",
        "--output",
        str(miss),
        "--manifest",
        str(manifest),
        "--cases",
        "IF-08",
        "--mpi-mode",
        "on",
    )
    assert skipped.returncode == 0, skipped.stderr
    missed = json.loads((miss / "plan.json").read_text(encoding="utf-8"))
    assert missed["jobs"] == []


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
    assert [(job["case_id"], job["pops_native_dim"]) for job in plan["jobs"]] == [
        ("CP-02", 1),
        ("CP-02", 2),
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
    assert [(job["case_id"], job["pops_native_dim"]) for job in plan["jobs"]] == [
        ("CP-02", 1)
    ]


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
    assert [(job["case_id"], job["pops_native_dim"]) for job in plan["jobs"]] == [
        ("CP-02", 1)
    ]


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


def _load_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_verification_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _honest_run_fields(*, n_cells: int = 32) -> dict:
    return {
        "compiler": "not-compiled",
        "build_type": "not-compiled",
        "precision": "not-compiled",
        "block_size": [n_cells],
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": "fixture",
        "cfl": 0.45,
        "final_time": 0.25,
        "gpus": 0,
    }


def test_invoke_refuses_n_cells_only_signature():
    from verification.pops_verify.campaign import CampaignJob, CampaignRequest

    runner = _load_runner()

    def run_native(n_cells=8, t_end=0.01):
        return n_cells

    request = CampaignRequest.from_job(
        CampaignJob(case_id="IF-08", pops_native_dim=1, min_resolution=32)
    )
    with pytest.raises(runner.VerificationRunnerError, match="request"):
        runner.invoke_run_native(run_native, request)


def test_real_if01_and_if08_run_native_accept_campaign_request():
    import inspect

    from verification.pops_verify.campaign import CampaignJob, CampaignRequest
    from verification.pops_verify.case_authoring import load_sibling_module

    runner = _load_runner()
    cases = (
        (
            REPO_ROOT / "verification/cases/infrastructure/mpi_invariance/run.py",
            "IF-01",
        ),
        (
            REPO_ROOT / "verification/cases/infrastructure/native_dim_guard/run.py",
            "IF-08",
        ),
    )
    for path, case_id in cases:
        module = load_sibling_module(path)
        assert "request" in inspect.signature(module.run_native).parameters, path
        request = CampaignRequest.from_job(
            CampaignJob(case_id=case_id, pops_native_dim=1, min_resolution=16)
        )
        try:
            runner.invoke_run_native(module.run_native, request)
        except TypeError as exc:
            raise AssertionError(f"{case_id} rejected CampaignRequest: {exc}") from exc
        except Exception as exc:
            assert exc.__class__.__name__ == "NativeUnavailable", (
                f"{case_id} raised {exc.__class__.__name__}: {exc}"
            )


def test_if08_run_native_with_dim1_request_dispatches_dim1(monkeypatch):
    from verification.pops_verify.campaign import CampaignJob, CampaignRequest
    from verification.pops_verify.case_authoring import load_sibling_module

    run = load_sibling_module(
        REPO_ROOT / "verification/cases/infrastructure/native_dim_guard/run.py"
    )
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    request = CampaignRequest.from_job(CampaignJob(case_id="IF-08", pops_native_dim=1))
    with pytest.raises(run.NativeUnavailable) as exc_info:
        run.run_native(request=request)
    assert "required dim 2" not in str(exc_info.value)


def test_if01_run_native_forwards_request_mpi_mode():
    source = (
        REPO_ROOT / "verification/cases/infrastructure/mpi_invariance/run.py"
    ).read_text(encoding="utf-8")
    assert "request" in source
    assert "mpi_mode=request.mpi_mode" in source or (
        "bind_public(" in source and "request.mpi_mode" in source
    )


def test_execute_refuses_invented_provenance_when_run_facts_unknown(tmp_path: Path):
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py="def run_native(request=None, n_cells=None):\n    return None\n",
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    job_dir = output / "IF-08" / "dim1-KokkosSerial-off"
    assert result.returncode == 1
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "fail"
    reason = (rows[0].get("reason") or "").lower()
    assert "provenance" in reason or "unknown" in reason
    if (job_dir / "provenance.json").is_file():
        provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
        raise AssertionError(f"invented provenance was written: {provenance}")


def test_execute_writes_honest_provenance_from_returned_run_fields(tmp_path: Path):
    from jsonschema import Draft202012Validator

    fields = _honest_run_fields(n_cells=32)
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py=(
            "def run_native(request=None, n_cells=None):\n"
            f"    return {fields!r}\n"
        ),
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 0, result.stderr
    job_dir = output / "IF-08" / "dim1-KokkosSerial-off"
    provenance = json.loads((job_dir / "provenance.json").read_text(encoding="utf-8"))
    Draft202012Validator(
        json.loads((REPO_ROOT / "schemas" / "verification_provenance.v1.json").read_text()),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(provenance)
    assert provenance["cfl"] == 0.45
    assert provenance["compiler"] == "not-compiled"
    assert provenance["precision"] == "not-compiled"
    assert provenance["final_time"] == 0.25
    assert provenance["kokkos_execution_space"] == "KokkosSerial"


def test_execute_writes_artifacts_only_on_rank_zero(tmp_path: Path):
    fields = _honest_run_fields(n_cells=32)
    manifest, _case_dir = _write_case_manifest(
        tmp_path,
        case_id="IF-08",
        run_py=(
            "def run_native(request=None, n_cells=None):\n"
            f"    return {fields!r}\n"
        ),
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={
            "POPS_NATIVE_DIM": "1",
            "POPS_NATIVE_VARIANTS_ROOT": str(leaf),
            "POPS_CAMPAIGN_RANK": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    assert not (output / "results.json").exists()
    assert not (output / "IF-08" / "dim1-KokkosSerial-off" / "resolved_case.json").exists()
    assert not (output / "summary.json").exists()


def test_execute_rejects_path_escaping_case_id(tmp_path: Path):
    case_dir = tmp_path / "cases" / "safe"
    case_dir.mkdir(parents=True)
    (case_dir / "run.py").write_text(
        "def run_native(request=None, n_cells=None):\n    return None\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_header()
        + f"""
[[case]]
id = "../ESC-01"
path = "{case_dir / "run.py"}"
name = "escaped case"
verification_kind = "infrastructure"
evidence_status = "required"
physics = []
oracle = "fixture"
native_dimensions = [1]
execution_spaces = ["KokkosSerial"]
mpi_modes = ["off"]
suites = ["pr"]
requires = []

[case.resources.pr]
nodes = 1
mpi_ranks = 1
omp_threads = 1
resolutions = [32]

[case.resources.two_node]
nodes = [1, 2]

[case.acceptance]
finite = true
""",
        encoding="utf-8",
    )
    leaf = _write_authenticated_leaf(tmp_path)
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
        env={"POPS_NATIVE_DIM": "1", "POPS_NATIVE_VARIANTS_ROOT": str(leaf)},
    )
    assert result.returncode == 1
    combined = f"{result.stderr}\n{result.stdout}".lower()
    assert "case" in combined or "path" in combined or "unsafe" in combined
    assert not (tmp_path / "ESC-01").exists()
    escaped = list(tmp_path.glob("**/provenance.json"))
    assert escaped == []


def test_case_module_path_resolves_against_repo_root(tmp_path: Path):
    runner = _load_runner()
    relative = "verification/cases/infrastructure/native_dim_guard/run.py"
    resolved = runner.resolve_case_module_path(relative)
    assert resolved == (REPO_ROOT / relative).resolve()
    monkeypatch_cwd = tmp_path
    import os

    previous = os.getcwd()
    os.chdir(monkeypatch_cwd)
    try:
        again = runner.resolve_case_module_path(relative)
    finally:
        os.chdir(previous)
    assert again == (REPO_ROOT / relative).resolve()


def test_execute_resolves_relative_case_path_from_foreign_cwd(tmp_path: Path):
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_header()
        + """
[[case]]
id = "IF-08"
path = "verification/cases/infrastructure/native_dim_guard/run.py"
name = "exact native-dim specialization"
verification_kind = "infrastructure"
evidence_status = "required"
physics = ["transport"]
oracle = "fixture"
native_dimensions = [1]
execution_spaces = ["KokkosSerial"]
mpi_modes = ["off"]
suites = ["pr"]
requires = []

[case.resources.pr]
nodes = 1
mpi_ranks = 1
omp_threads = 1
resolutions = [16]

[case.resources.two_node]
nodes = [1, 2]

[case.acceptance]
finite = true
""",
        encoding="utf-8",
    )
    leaf = _write_authenticated_leaf(tmp_path)
    output = tmp_path / "out"
    env = os.environ.copy()
    env.pop("POPS_NATIVE_DIM", None)
    env["POPS_NATIVE_DIM"] = "1"
    env["POPS_NATIVE_VARIANTS_ROOT"] = str(leaf)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'python'}{os.pathsep}{REPO_ROOT}"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    combined = f"{result.stderr}\n{result.stdout}"
    assert "No such file" not in combined
    assert "cannot load" not in combined.lower()
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert rows[0]["case_id"] == "IF-08"
    assert "run.py" in rows[0].get("path", "")
