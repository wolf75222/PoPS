"""IF-08 exact native-dim specialization (planner refuse; no live native artifact)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.campaign import (
    CampaignError,
    CampaignJob,
    expand_jobs,
)
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "infrastructure" / "native_dim_guard"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
ORDERS_REASON = "exact native-dim specialization / no live native artifact"


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_call_owners(source: str) -> list[str]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    owners: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pops"
        ):
            continue
        current: ast.AST | None = node
        owner = "<module>"
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = current.name
                break
            current = parents.get(current)
        owners.append(owner)
    return owners


def test_expand_jobs_refuses_requested_dim_that_differs_from_artifact():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "expand_jobs" in text
    assert "resolve_artifact_dim" in text
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    with pytest.raises(CampaignError, match="POPS_NATIVE_DIM") as exc_info:
        expand_jobs([exact.tr01_case()], [2], artifact_dim=1)
    assert "fallback" in str(exc_info.value).lower()
    with pytest.raises(CampaignError, match="POPS_NATIVE_DIM"):
        run.plan_tr01_jobs([2], artifact_dim=1)


def test_matching_dim_emits_tr01():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    expected = [CampaignJob(case_id="TR-01", pops_native_dim=1)]
    assert expand_jobs([exact.tr01_case()], [1], artifact_dim=1) == expected
    assert run.plan_tr01_jobs([1], artifact_dim=1) == expected
    assert (
        run.plan_tr01_jobs([1], artifact_dim=None, environ={"POPS_NATIVE_DIM": "1"})
        == expected
    )


def test_mismatch_is_refused_before_fake_run():
    run = _load_case_module("run")
    run.reset_fake_runs()
    with pytest.raises(CampaignError, match="POPS_NATIVE_DIM"):
        run.fake_run_tr01([2], artifact_dim=1)
    assert run.fake_run_count() == 0
    result = run.fake_run_tr01([1], artifact_dim=1)
    assert run.fake_run_count() == 1
    assert result["jobs"] == [CampaignJob(case_id="TR-01", pops_native_dim=1)]
    assert result["ran"] is True


def test_write_if08_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_if08_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"] == ORDERS_REASON


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


def test_run_doctor_wraps_pops_doctor():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "def run_doctor" in text
    assert "pops.doctor" in text
    run = _load_case_module("run")
    try:
        report = run.run_doctor()
    except run.NativeUnavailable as exc:
        assert "pops.doctor" in str(exc)
        return
    assert isinstance(report, dict)
    assert report


def test_manifest_if08_is_selectable_at_dim2_serial():
    """The Dim2 Serial campaign job must be able to expand IF-08, not plan zero jobs."""
    import tomllib

    from verification.pops_verify.campaign import expand_jobs

    manifest = tomllib.loads(
        (REPO_ROOT / "verification" / "manifest.toml").read_text(encoding="utf-8")
    )
    case = next(item for item in manifest["case"] if item["id"] == "IF-08")
    assert 2 in case["native_dimensions"]
    jobs = expand_jobs(
        [case],
        [2],
        artifact_dim=2,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
    )
    assert [job.case_id for job in jobs] == ["IF-08"]
    assert jobs[0].pops_native_dim == 2
    assert jobs[0].execution_space == "KokkosSerial"
    assert jobs[0].mpi_mode == "off"


def test_if08_dim2_request_returns_run_fields(monkeypatch):
    """A successful Dim2 campaign path must return provenance fields, not a raw array."""
    from verification.pops_verify.campaign import CampaignJob, CampaignRequest
    from verification.pops_verify.provenance import RUN_FIELDS

    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")

    class _Ge03:
        CFL = 0.4

        @staticmethod
        def run_native(n_cells=8, t_end=0.01):
            return [n_cells, t_end]

    monkeypatch.setattr(run, "load_sibling_module", lambda path: _Ge03())
    request = CampaignRequest.from_job(
        CampaignJob(case_id="IF-08", pops_native_dim=2, min_resolution=8)
    )
    result = run.run_native(request=request)
    assert isinstance(result, dict)
    missing = [key for key in RUN_FIELDS if key not in result]
    assert missing == []
    assert result["cfl"] == _Ge03.CFL
    assert result["final_time"] == 0.01
    assert result["resolution"] == [8, 8]


def test_dim2_case_under_dim1_raises_before_fake_or_native_run(monkeypatch):
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    assert exact.dim2_case()["id"] == "GE-03"
    assert list(exact.dim2_case()["native_dimensions"]) == [2]
    run.reset_fake_runs()
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM") as exc_info:
        run.present_dim2_case(artifact_dim=1)
    assert "fallback" in str(exc_info.value).lower()
    assert run.fake_run_count() == 0
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.fake_run_dim2(artifact_dim=1)
    assert run.fake_run_count() == 0
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.fake_run_dim2(environ={"POPS_NATIVE_DIM": "1"})
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.present_dim1_case(artifact_dim=2)
    assert run.fake_run_count() == 0
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.run_native(8, t_end=0.01)
    assert run.fake_run_count() == 0
