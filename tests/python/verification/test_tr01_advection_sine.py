"""TR-01 3-d oblique periodic advection sine (Annexe A.1 / §35.1)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.cell_averages import analytic_cell_averages
from verification.pops_verify.convergence import observed_order
from verification.pops_verify.reference_errors import reference_errors
from verification.pops_verify.report import ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "transport" / "advection_sine"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
RESOLUTIONS = (16, 32, 64, 128)


def _load_case_module(name: str):
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tr01_advection_sine_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_canonical_3d_data_matches_annexe_a():
    exact = _load_case_module("exact")
    assert exact.REQUIRED_NATIVE_DIM == 3
    assert tuple(exact.A) == (1.0, 1.0, 1.0)
    assert tuple(exact.K) == (1.0, 2.0, 3.0)
    assert float(exact.T_END) == 1.0
    assert tuple(exact.RESOLUTIONS) == RESOLUTIONS
    xx, yy, zz, volumes = exact.uniform_cell_mesh(8)
    q0 = exact.exact_sine(xx, yy, zz, 0.0)
    q1 = exact.exact_sine(xx, yy, zz, 1.0)
    np.testing.assert_allclose(q0, q1, atol=1.0e-12)
    assert volumes.shape == (8, 8, 8)
    assert xx.shape == (8, 8, 8)


def test_reference_errors_of_exact_vs_exact_are_zero():
    exact = _load_case_module("exact")
    xx, yy, zz, volumes = exact.uniform_cell_mesh(8)
    field = exact.exact_sine(xx, yy, zz, 0.0)
    errors = reference_errors(field, field, volumes)
    assert errors.l1 == 0.0
    assert errors.l2 == 0.0
    assert errors.linf == 0.0


def test_cell_average_oracle_is_finite_on_the_cube():
    exact = _load_case_module("exact")
    lo, hi = exact.cell_bounds(8)

    def _u(x, y, z, time):
        return exact.exact_sine(x, y, z, time)

    averages = analytic_cell_averages(_u, lo, hi, 0.0)
    assert averages.shape == (8, 8, 8)
    assert np.isfinite(averages).all()


def test_manufactured_second_order_series_observed_order_is_two(tmp_path: Path):
    n = np.asarray(RESOLUTIONS, dtype=np.float64)
    spacings = 1.0 / n
    errors = spacings**2
    orders = observed_order(errors, spacings)
    np.testing.assert_allclose(orders, np.full(orders.shape, 2.0))

    analyze = _load_case_module("analyze")
    analyze.analyze_series(errors, spacings, tmp_path)
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    observed = [row["observed_order"] for row in loaded["orders"]]
    assert observed
    np.testing.assert_allclose(observed, np.full(len(observed), 2.0))
    assert loaded["native_dimensions"] == [3]


def test_build_case_and_resolve_plan_are_dimension_3():
    run = _load_case_module("run")
    case = run.build_case(8)
    plan = run.resolve_plan(8)
    assert case is not None
    assert getattr(plan, "resolved_dimension", None) == 3
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "Cartesian3D" in text
    assert "POPS_NATIVE_DIM=3" in text or "REQUIRED_NATIVE_DIM" in text
    assert "Cartesian1D" not in text


def test_run_native_refuses_missing_or_non_three_dim(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM=3"):
        run._require_native_dim3()
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    with pytest.raises(run.NativeUnavailable, match="no 1-d/2-d fallback"):
        run._require_native_dim3()
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    with pytest.raises(run.NativeUnavailable, match="no 1-d/2-d fallback"):
        run._require_native_dim3()


def test_write_tr01_report_writes_four_schema_valid_artifacts(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_tr01_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["native_dimensions"] == [3]


def test_modules_do_not_hardcode_pops_run_except_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        owners = _pops_run_call_owners(text)
        if name == "run.py":
            assert owners
            assert set(owners) <= {"run_native", "run_order_campaign"}
        else:
            assert owners == []
            assert "pops.run(" not in text


@pytest.mark.compiler
def test_run_native_dim3_returns_cube_or_skips(tmp_path: Path):
    run = _load_case_module("run")
    missing = missing_compiler_requirement()
    try:
        field = np.asarray(
            run.run_native(8, t_end=0.05, output_dir=tmp_path), dtype=np.float64
        )
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert field.shape == (8, 8, 8)
    assert np.isfinite(field).all()
    assert (tmp_path / "provenance.json").is_file()
    document = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert document["schema"] == "pops.verification.provenance.v1"
    assert document["pops_native_dim"] == 3
    assert document["dimension"] == 3
    assert document["resolution"] == [8, 8, 8]


@pytest.mark.compiler
def test_order_campaign_requires_four_resolutions():
    run = _load_case_module("run")
    with pytest.raises(ValueError, match="four resolutions"):
        run.run_order_campaign((16, 32, 64))
