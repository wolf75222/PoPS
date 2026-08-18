"""RB-07 Liska–Wendroff implosion (in-memory IC + diagonal leftover)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from tests.python.support.requirements import missing_compiler_requirement
from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.report import ARTIFACTS
from verification.pops_verify.symmetry import xy_symmetry_error

REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = REPO_ROOT / "verification" / "cases" / "robustness" / "liska_implosion"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_report.v1.json"
CASE_MODULES = ("exact.py", "run.py", "analyze.py")
N_CELLS = 32


def _load_case_module(name: str):
    return load_sibling_module(CASE_DIR / f"{name}.py")


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _cell_mesh(exact, n_cells: int = N_CELLS):
    length = float(exact.DOMAIN_UPPER[0] - exact.DOMAIN_LOWER[0])
    width = length / float(n_cells)
    origin = float(exact.DOMAIN_LOWER[0])
    centers = origin + (np.arange(n_cells, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y


def test_ic_jump_on_x_plus_y_equals_015():
    exact = _load_case_module("exact")
    assert exact.GAMMA == 1.4
    assert exact.DIAGONAL == 0.15
    assert exact.RHO_OUT == 1.0
    assert exact.P_OUT == 1.0
    assert exact.RHO_IN == 0.125
    assert exact.P_IN == 0.14
    assert exact.DOMAIN_LOWER == (0.0, 0.0)
    assert exact.DOMAIN_UPPER == (0.3, 0.3)
    sums = np.array([0.05, 0.10, 0.14, 0.15, 0.16, 0.25, 0.40])
    x = 0.5 * sums
    y = 0.5 * sums
    primitives = exact.primitives(x, y, 0.0)
    inside = sums <= exact.DIAGONAL
    outside = sums > exact.DIAGONAL
    assert np.any(inside) and np.any(outside)
    np.testing.assert_allclose(primitives["rho"][inside], exact.RHO_IN)
    np.testing.assert_allclose(primitives["p"][inside], exact.P_IN)
    np.testing.assert_allclose(primitives["rho"][outside], exact.RHO_OUT)
    np.testing.assert_allclose(primitives["p"][outside], exact.P_OUT)
    np.testing.assert_allclose(primitives["u"], 0.0)
    np.testing.assert_allclose(primitives["v"], 0.0)
    on_cut = exact.primitives(0.075, 0.075, 0.0)
    np.testing.assert_allclose(on_cut["rho"], exact.RHO_IN)
    np.testing.assert_allclose(on_cut["p"], exact.P_IN)


def test_exact_ic_is_symmetric_under_xy_swap():
    exact = _load_case_module("exact")
    x = np.array([0.02, 0.08, 0.20, 0.25])
    y = np.array([0.10, 0.22, 0.05, 0.04])
    state = exact.primitives(x, y, 0.0)
    swapped = exact.primitives(y, x, 0.0)
    np.testing.assert_allclose(state["rho"], swapped["rho"])
    np.testing.assert_allclose(state["p"], swapped["p"])
    np.testing.assert_allclose(state["u"], swapped["v"])
    np.testing.assert_allclose(state["v"], swapped["u"])
    mesh_x, mesh_y = _cell_mesh(exact)
    field = exact.primitives(mesh_x, mesh_y, 0.0)
    reflected = exact.reflect(field)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_allclose(field[key], reflected[key])
    np.testing.assert_allclose(xy_symmetry_error(field["rho"]), 0.0)
    np.testing.assert_allclose(xy_symmetry_error(field["p"]), 0.0)


def test_leftover_residual_is_zero_on_exact_field():
    exact = _load_case_module("exact")
    run = _load_case_module("run")
    mesh_x, mesh_y = _cell_mesh(exact)
    field = exact.primitives(mesh_x, mesh_y, 0.0)
    residual = exact.leftover_residual(field)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_allclose(residual[key], 0.0)
    run_residual = run.leftover_residual(field)
    for key in ("rho", "u", "v", "p"):
        np.testing.assert_allclose(run_residual[key], residual[key])
    broken = {key: np.array(value, copy=True) for key, value in field.items()}
    broken["rho"][0, -1] = broken["rho"][0, -1] + 1.0
    broken_residual = exact.leftover_residual(broken)
    assert np.max(np.abs(broken_residual["rho"])) > 0.0


def test_write_rb07_report_writes_four_artifacts_and_schema(tmp_path: Path):
    analyze = _load_case_module("analyze")
    written = analyze.write_rb07_report(tmp_path)
    assert written == ARTIFACTS
    for name in ARTIFACTS.values():
        assert (tmp_path / name).is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    _validator().validate(loaded)
    assert loaded["schema"] == "pops.verification.report.v1"
    assert loaded["orders"] == []
    assert loaded["native_dimensions"] == [2]
    reason = loaded["not_applicable_reason"]["orders"]
    assert "implosion / no analytic late-time" in reason


def test_modules_use_load_sibling_module_not_from_exact_import():
    for name in ("run.py", "analyze.py"):
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "load_sibling_module" in text
        assert "from exact import" not in text
        assert "from run import" not in text


def _source_without_run_native(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    skip: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_native":
            last = getattr(node, "end_lineno", node.lineno)
            skip.update(range(node.lineno, last + 1))
    return "".join(line for index, line in enumerate(lines, start=1) if index not in skip)


def test_no_pops_run_outside_run_native():
    for name in CASE_MODULES:
        text = (CASE_DIR / name).read_text(encoding="utf-8")
        if name == "run.py":
            assert "pops.run" not in _source_without_run_native(text)
        else:
            assert "pops.run" not in text


def test_resolve_plan_returns_without_authoring_pending():
    run = _load_case_module("run")
    plan = run.resolve_plan(16)
    assert plan is not None
    assert getattr(plan, "resolved_dimension", None) == 2


def test_authoring_uses_public_slip_wall_on_all_faces():
    text = (CASE_DIR / "run.py").read_text(encoding="utf-8")
    assert "SlipWall" in text
    assert "TransportBoundarySet" in text
    assert "uniform_periodic_layout" not in text
    assert "frame.boundaries.all" in text


def test_run_native_refuses_non_dim2(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "1")
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM") as exc_info:
        run.run_native(8, t_end=0.01)
    message = str(exc_info.value)
    assert "fallback" in message.lower()
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)
    with pytest.raises(run.NativeUnavailable, match="POPS_NATIVE_DIM"):
        run.run_native(8, t_end=0.01)


@pytest.mark.compiler
def test_run_native_returns_finite_conserved_or_skips(monkeypatch):
    run = _load_case_module("run")
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    missing = missing_compiler_requirement()
    try:
        conserved = run.run_native(16, t_end=0.05)
    except run.NativeUnavailable as exc:
        if missing:
            pytest.skip(missing)
        pytest.skip(str(exc))
    assert set(conserved) == set(run.COMPONENT_ORDER)
    for field in conserved.values():
        array = np.asarray(field, dtype=np.float64)
        assert array.shape == (16, 16)
        assert np.isfinite(array).all()
    primitives = run.conserved_to_primitives(conserved)
    assert np.all(primitives["rho"] > 0.0)
    assert np.all(primitives["p"] > 0.0)
