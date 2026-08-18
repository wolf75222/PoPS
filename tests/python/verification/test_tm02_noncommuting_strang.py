"""TM-02 noncommuting Strang (in-memory A/B split plus public Strang Case)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "time" / "noncommuting_strang"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
DT = 0.1
DT_SERIES = tuple(DT / factor for factor in (1, 2, 4, 8))
LIE_THRESHOLD = 0.8
STRANG_THRESHOLD = 1.8


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


def test_operators_do_not_commute():
    text = (CASE_DIR / "exact.py").read_text(encoding="utf-8")
    assert "from exact import" not in text
    exact = _load_case_module("exact")
    assert exact.A1 != exact.A2
    operator_a = np.asarray(exact.operator_A(), dtype=np.float64)
    operator_b = np.asarray(exact.operator_B(), dtype=np.float64)
    expected_a = np.diag([exact.A1, exact.A2])
    expected_b = np.array(
        [[-exact.NU, exact.NU], [exact.NU, -exact.NU]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(operator_a, expected_a)
    np.testing.assert_allclose(operator_b, expected_b)
    commutator = operator_a @ operator_b - operator_b @ operator_a
    assert not np.allclose(commutator, 0.0)
    np.testing.assert_allclose(exact.commutator(), commutator)
    assert exact.operators_commute() is False


def test_utility_matrix_lie_order_is_not_scientific_evidence():
    run = _load_case_module("run")
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    dts = np.asarray(DT_SERIES, dtype=np.float64)
    errors = np.asarray(run.error_series(run.lie_step, dts), dtype=np.float64)
    orders = observed_order(errors, dts)
    assert np.all(orders > LIE_THRESHOLD)
    assert np.all(orders < 1.3)
    np.testing.assert_allclose(orders, np.ones(orders.shape), atol=0.05)


def test_utility_matrix_strang_order_is_not_scientific_evidence():
    run = _load_case_module("run")
    dts = np.asarray(DT_SERIES, dtype=np.float64)
    errors = np.asarray(run.error_series(run.strang_step, dts), dtype=np.float64)
    orders = observed_order(errors, dts)
    assert np.all(orders > STRANG_THRESHOLD)
    assert np.all(orders < 2.2)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0), atol=0.05)


def test_write_tm02_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    text = (CASE_DIR / "analyze.py").read_text(encoding="utf-8")
    assert "load_sibling_module" in text
    assert "from exact import" not in text
    written = analyze.write_tm02_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["coverage"]["cases_passed"] == 0
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]


def test_build_case_and_resolve_plan_without_native():
    run = _load_case_module("run")
    case = run.build_case(DT)
    plan = run.resolve_plan(DT)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 1
    assert run.N_CELLS >= 16
    one_cell = run.resolve_plan(DT, n_cells=1)
    spatial = run.resolve_plan(DT, n_cells=run.N_CELLS)
    assert getattr(one_cell, "resolved_dimension", None) == 1
    assert getattr(spatial, "resolved_dimension", None) == 1


def test_public_case_uses_official_strang_and_local_linear_maps():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert 'components=("q0", "q1")' in text
    assert 'components=("u", "v")' not in text
    assert "libtime.Strang" in text
    assert "libtime.Lie" in text
    assert "local_linear_operator" in text
    assert "StrangFV" not in text
    assert "LieFV" not in text
    assert "_heun_step" not in text
    assert "_strang_fv_program" not in text
    run = _load_case_module("run")
    plan = run.resolve_plan(DT)
    assert getattr(plan, "resolved_dimension", None) == 1


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert set(owners) <= {"run_native"}
        else:
            assert owners == []
            assert "pops.run(" not in text


@pytest.mark.compiler
def test_compiler_run_native_returns_field_when_kokkos_present():
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(
            run.run_native(DT, t_end=0.05, n_cells=16), dtype=np.float64
        )
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.size == 2 * 16
    assert np.isfinite(field).all()


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="TM-02",
        pops_native_dim=1,
        suite="pr",
        execution_space="KokkosSerial",
        mpi_mode="off",
        min_resolution=8,
        resources=CampaignResources(resolutions=(8,)),
        evidence_status="required",
    )
    try:
        result = run.run_native(request=request)
    except run.NativeUnavailable:
        return
    assert isinstance(result, dict)
    assert "resolution" in result
    assert "result" in result


def test_report_fails_closed_without_native_output(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_tm02_report(tmp_path)
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
    assert (
        loaded["coverage"]["cases_failed"] + loaded["coverage"]["cases_not_supported"]
        >= 1
    )
    reasons = " ".join(item["reason"] for item in loaded["failures"])
    notes = " ".join(loaded["coverage"].get("not_tested") or [])
    blob = (reasons + " " + notes).lower()
    assert (
        "native" in blob
        or "kokkos" in blob
        or "supported" in blob
        or "required" in blob
        or "not " in blob
        or "no " in blob
    )


def test_analyze_native_requires_native_field():
    analyze = _load_case_module("analyze")
    try:
        analyze.analyze_native({})
    except (ValueError, TypeError, KeyError):
        return
    raise AssertionError("analyze_native must refuse an empty mapping")


def test_analyze_native_computes_field_errors():
    analyze = _load_case_module("analyze")
    result = analyze.analyze_native(
        {
            "field": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "oracle": np.array([1.0, 2.0, 2.5], dtype=np.float64),
            "volumes": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        }
    )
    assert result["linf"] == 0.5
    assert result["l1"] > 0.0
    assert result["l2"] > 0.0


def test_write_report_stays_fail_closed_with_native_mapping(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_tm02_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0


def test_analyze_native_computes_orders_from_native_series():
    analyze = _load_case_module("analyze")
    dts = (0.1, 0.05, 0.025)
    lie = (0.1, 0.05, 0.025)
    strang = (0.1, 0.025, 0.00625)
    result = analyze.analyze_native(
        {
            "lie": {"linf": lie, "dts": dts},
            "strang": {"linf": strang, "dts": dts},
        }
    )
    np.testing.assert_allclose(result["lie_orders"], observed_order(lie, dts))
    np.testing.assert_allclose(result["strang_orders"], observed_order(strang, dts))
