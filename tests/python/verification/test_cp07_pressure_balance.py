"""CP-07 pressure–field equilibrium (in-memory isothermal oracle; no solver)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "euler_poisson" / "pressure_balance"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 64
FORCE_ATOL = 1.0e-12


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _pops_run_call_outside_run_native(text: str) -> bool:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "run"):
            continue
        value = func.value
        if not (isinstance(value, ast.Name) and value.id == "pops"):
            continue
        if _enclosing_function(tree, node) != "run_native":
            return True
    return False


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str | None:
    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []
            self.found: str | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def generic_visit(self, node: ast.AST) -> None:
            if node is target and self.stack:
                self.found = self.stack[-1]
            super().generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.found


def _assert_force_balance(fields: dict) -> None:
    density = np.asarray(fields["n"], dtype=np.float64)
    electric = np.asarray(fields["E"], dtype=np.float64)
    grad_p = np.asarray(fields["grad_p"], dtype=np.float64)
    charge = float(fields["q"])
    np.testing.assert_allclose(
        grad_p,
        charge * density * electric,
        rtol=0.0,
        atol=FORCE_ATOL,
    )


def test_cosine_force_balance_grad_p_equals_q_n_E():
    exact = _load_case_module("exact")
    x, _ = exact.uniform_cell_centers(N_CELLS)
    fields = exact.exact_fields(x, profile="cosine")
    assert float(exact.DELTA) < 1.0
    assert np.all(np.asarray(fields["n"], dtype=np.float64) > 0.0)
    _assert_force_balance(fields)


def test_gaussian_force_balance_grad_p_equals_q_n_E():
    exact = _load_case_module("exact")
    x, _ = exact.uniform_cell_centers(N_CELLS, x_lo=-0.5, x_hi=0.5)
    fields = exact.exact_fields(x, profile="gaussian")
    assert np.all(np.asarray(fields["n"], dtype=np.float64) > 0.0)
    _assert_force_balance(fields)


def test_velocity_is_identically_zero():
    exact = _load_case_module("exact")
    x_cos, _ = exact.uniform_cell_centers(N_CELLS)
    x_gauss, _ = exact.uniform_cell_centers(N_CELLS, x_lo=-0.5, x_hi=0.5)
    for x, profile in ((x_cos, "cosine"), (x_gauss, "gaussian")):
        fields = exact.exact_fields(x, profile=profile)
        velocity = np.asarray(fields["u"], dtype=np.float64)
        np.testing.assert_allclose(velocity, 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            exact.u_exact(x),
            0.0,
            rtol=0.0,
            atol=0.0,
        )


def test_boltzmann_potential_and_minus_grad_phi():
    exact = _load_case_module("exact")
    x, _ = exact.uniform_cell_centers(N_CELLS)
    fields = exact.exact_fields(x, profile="cosine")
    density = np.asarray(fields["n"], dtype=np.float64)
    temperature = float(fields["T"])
    charge = float(fields["q"])
    offset = float(fields["C"])
    expected_phi = -(temperature / charge) * np.log(density) + offset
    np.testing.assert_allclose(
        fields["phi"],
        expected_phi,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        fields["E"],
        exact.e_from_phi(x, fields["phi"]),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_write_cp07_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_cp07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["not_applicable_reason"]["orders"]
    assert loaded["coverage"]["cases_passed"] == 0


def test_case_modules_use_load_sibling_module_and_omit_pops_run():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert _pops_run_call_outside_run_native(text) is False
        assert "pops.run(" not in text or "def run_native" in text
        assert "from exact import" not in text
        if name != "exact.py":
            assert "load_sibling_module" in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None


def test_background_charge_makes_poisson_identity():
    run = _load_case_module("run")
    fields = run.build_oracle(32, profile="cosine")
    spacing = 1.0 / 32.0
    lap_phi = run._spectral_laplacian(fields["phi"], spacing)
    rhs = (float(fields["q"]) * fields["n"] + run.background_charge(32)) / run.EPS0
    np.testing.assert_allclose(-lap_phi, rhs, rtol=0.0, atol=1.0e-10)


@pytest.mark.compiler
def test_compiler_run_native_returns_field_when_kokkos_present():
    import pytest

    from tests.python.support.requirements import missing_compiler_requirement

    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(run.run_native(16, t_end=0.02), dtype=np.float64)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (3, 16)
    assert np.isfinite(field).all()
    assert np.all(field[0] > 0.0)


def test_run_native_accepts_campaign_request():
    import inspect
    from verification.pops_verify.campaign import CampaignRequest, CampaignResources

    run = _load_case_module("run")
    assert "request" in inspect.signature(run.run_native).parameters
    request = CampaignRequest(
        case_id="CP-07",
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
    written = analyze.write_cp07_report(tmp_path)
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
    written = analyze.write_cp07_report(
        tmp_path,
        native={"field": np.array([1.0, 2.0], dtype=np.float64)},
    )
    assert written == ARTIFACTS
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["coverage"]["cases_passed"] == 0
